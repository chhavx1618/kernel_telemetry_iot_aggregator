"""Asynchronous task queue for handling inference requests."""
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any, List

import redis
from rq import Queue as RQQueue, Retry
from rq.job import Job

from ..models import Database, InferenceResult
from ..config import Config

logger = logging.getLogger(__name__)

class AsyncQueue:
    """Manages asynchronous processing of inference requests."""
    
    def __init__(self, engine: Any, db: Database, redis_client: Optional[redis.Redis] = None):
        """Initialize the async queue.
        
        Args:
            engine: Inference engine instance
            db: Database instance for logging
            redis_client: Optional Redis client (will be created if not provided)
        """
        self.engine = engine
        self.db = db
        self.redis_client = redis_client or self._connect_redis()
        self.queue = None
        self.worker_process = None
        
        if self.redis_client is not None:
            self.queue = RQQueue(connection=self.redis_client)
    
    def _connect_redis(self) -> Optional[redis.Redis]:
        """Connect to Redis server."""
        try:
            return redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None
    
    def enqueue_inference(
        self, 
        image_path: Path, 
        callback_url: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Enqueue an inference request.
        
        Args:
            image_path: Path to the image file
            callback_url: Optional URL to send results to when complete
            metadata: Additional metadata to include with the job
            
        Returns:
            Job ID if successful, None otherwise
        """
        if not self.queue:
            logger.error("Cannot enqueue job: Redis queue not available")
            return None
            
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            job_meta = {
                'image_path': str(image_path.absolute()),
                'callback_url': callback_url,
                'submitted_at': time.time(),
                'metadata': metadata or {}
            }
            
            # Enqueue the job with retry on failure
            job = self.queue.enqueue(
                process_inference_job,
                job_meta,
                job_timeout=300,  # 5 minutes
                retry=Retry(max=3, interval=[10, 30, 60]),
                result_ttl=86400,  # Keep results for 24 hours
                ttl=3600,  # Don't run if not started within 1 hour
                failure_ttl=86400  # Keep failed jobs for 24 hours
            )
            
            logger.info(f"Enqueued inference job {job.id} for {image_path}")
            return job.id
            
        except Exception as e:
            logger.error(f"Failed to enqueue inference job: {e}", exc_info=True)
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Dictionary with job status and results if available, None if job not found
        """
        if not self.redis_client:
            return None
            
        try:
            job = Job.fetch(job_id, connection=self.redis_client)
            
            result = {
                'id': job.id,
                'status': job.get_status(),
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'enqueued_at': job.enqueued_at.isoformat() if job.enqueued_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'ended_at': job.ended_at.isoformat() if job.ended_at else None,
                'result': job.result,
                'exc_info': job.exc_info,
                'meta': job.meta
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None
    
    def process_job_sync(self, job_data: Dict) -> Dict:
        """Process a job synchronously (for worker processes).
        
        Args:
            job_data: Dictionary containing job data
            
        Returns:
            Dictionary with job results
        """
        try:
            image_path = Path(job_data['image_path'])
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Run inference
            start_time = time.time()
            result = self.engine.predict(image_path)
            processing_time = time.time() - start_time
            
            if result is None:
                raise RuntimeError("Inference failed")
            
            # Generate visualization
            output_dir = Config.OUTPUT_DIR / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{image_path.stem}_result{image_path.suffix}"
            self.engine.visualize_detections(image_path, result, output_path)
            
            # Prepare response
            response = {
                'job_id': job_data.get('job_id'),
                'status': 'completed',
                'processing_time': processing_time,
                'model_version': result.model_version,
                'timestamp': result.timestamp,
                'detections': result.detections,
                'confidence_scores': result.confidence_scores,
                'inference_time': result.inference_time,
                'output_path': str(output_path) if output_path.exists() else None,
                'metadata': job_data.get('metadata', {})
            }
            
            # Log the inference
            self.db.log_inference(result)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing job: {e}", exc_info=True)
            raise

def process_inference_job(job_data: Dict) -> Dict:
    """Process an inference job (runs in worker process).
    
    This function is called by RQ workers to process jobs.
    
    Args:
        job_data: Dictionary containing job data
        
    Returns:
        Dictionary with job results
    """
    # Import here to avoid circular imports
    from ..app import create_app, db, engine
    
    # Create a Flask app context for database access
    app = create_app()
    
    with app.app_context():
        # Initialize services
        queue = AsyncQueue(engine=engine, db=db)
        
        # Process the job
        result = queue.process_job_sync(job_data)
        
        # If a callback URL was provided, send the results
        callback_url = job_data.get('callback_url')
        if callback_url:
            try:
                import requests
                requests.post(callback_url, json=result, timeout=10)
            except Exception as e:
                logger.error(f"Failed to send callback to {callback_url}: {e}")
        
        return result
