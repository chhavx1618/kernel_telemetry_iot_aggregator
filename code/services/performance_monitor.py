"""Performance monitoring and drift detection service."""
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Gauge, Counter, Histogram

from ..models import Database, PerformanceMetrics, ModelMetadata, InferenceResult
from ..config import Config

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors model performance and detects data drift."""
    
    def __init__(self, db: Database, check_interval: int = 300):
        """Initialize the performance monitor.
        
        Args:
            db: Database instance
            check_interval: Interval in seconds between performance checks
        """
        self.db = db
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
        # Initialize metrics
        self.drift_score = Gauge(
            'pcb_drift_score', 
            'Data drift score for the active model',
            ['model_version']
        )
        
        self.accuracy_gauge = Gauge(
            'pcb_model_accuracy',
            'Current accuracy of the model',
            ['model_version']
        )
        
        self.inference_time_gauge = Gauge(
            'pcb_avg_inference_time',
            'Average inference time in seconds',
            ['model_version']
        )
        
        self.detection_rate_gauge = Gauge(
            'pcb_detection_rate',
            'Average number of detections per inference',
            ['model_version']
        )
    
    def start(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            logger.warning("Performance monitor is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Started performance monitoring service")
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Stopped performance monitoring service")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_performance()
                self._detect_drift()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def _check_performance(self) -> None:
        """Check model performance and update metrics."""
        active_model = self.db.get_active_model_metadata()
        if not active_model:
            logger.warning("No active model for performance check")
            return
        
        # Get recent inferences (last hour)
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT inference_time, json_array_length(detections) as detections_count
            FROM inferences
            WHERE model_version = ? AND timestamp >= ?
        ''', (active_model.version, cutoff))
        
        results = cursor.fetchall()
        
        if not results:
            logger.debug("No recent inferences for performance check")
            return
        
        # Calculate metrics
        inference_times = [r['inference_time'] for r in results]
        detection_counts = [r['detections_count'] for r in results]
        
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        avg_detection_count = np.mean(detection_counts) if detection_counts else 0
        
        # Update metrics
        self.inference_time_gauge.labels(model_version=active_model.version).set(avg_inference_time)
        self.detection_rate_gauge.labels(model_version=active_model.version).set(avg_detection_count)
        
        logger.debug(f"Performance metrics - Inference time: {avg_inference_time:.4f}s, "
                   f"Avg detections: {avg_detection_count:.2f}")
        
        # Log metrics to database
        metrics = PerformanceMetrics(
            model_version=active_model.version,
            timestamp=datetime.now().isoformat(),
            accuracy=active_model.accuracy,
            avg_confidence=0.0,  # Would be calculated from ground truth if available
            avg_inference_time=avg_inference_time,
            detection_count=int(avg_detection_count * len(results))
        )
        
        self.db.log_metrics(metrics)
    
    def _detect_drift(self) -> None:
        """Detect data drift in model predictions."""
        active_model = self.db.get_active_model_metadata()
        if not active_model:
            return
        
        # Get recent confidence scores
        window_size = min(Config.DRIFT_DETECTION_WINDOW, 1000)  # Limit window size
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT confidence_scores 
            FROM inferences 
            WHERE model_version = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (active_model.version, window_size))
        
        results = cursor.fetchall()
        
        if len(results) < window_size // 2:  # Need sufficient data
            logger.debug("Insufficient data for drift detection")
            return
        
        # Calculate drift score (simplified example - would use more sophisticated methods in production)
        all_confidences = []
        for row in results:
            confidences = json.loads(row['confidence_scores'])
            all_confidences.extend(confidences)
        
        if not all_confidences:
            return
        
        # Simple drift detection based on confidence score distribution
        mean_confidence = np.mean(all_confidences)
        std_confidence = np.std(all_confidences)
        
        # Calculate drift score (0-1, higher means more drift)
        # This is a simplified example - in practice, you'd use more sophisticated methods
        drift_score = 0.0
        if mean_confidence < 0.7:  # Adjust threshold as needed
            drift_score = min(1.0, (0.7 - mean_confidence) * 5)  # Scale to 0-1
        
        self.drift_score.labels(model_version=active_model.version).set(drift_score)
        
        logger.info(f"Drift detection - Score: {drift_score:.3f}, "
                   f"Mean confidence: {mean_confidence:.3f}±{std_confidence:.3f}")
        
        # Check if we need to trigger a rollback
        if drift_score > 0.7:  # Threshold for rollback
            self._trigger_rollback(
                active_model,
                f"High drift detected (score: {drift_score:.3f})"
            )
    
    def _trigger_rollback(self, current_model: ModelMetadata, reason: str) -> None:
        """Trigger a rollback to a previous model version if available."""
        logger.warning(f"Considering rollback due to: {reason}")
        
        # Get all model versions sorted by version (newest first)
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version, accuracy, avg_inference_time 
            FROM models 
            WHERE version != ?
            ORDER BY version DESC
        ''', (current_model.version,))
        
        previous_models = cursor.fetchall()
        
        if not previous_models:
            logger.warning("No previous model versions available for rollback")
            return
        
        # Find the most recent model with acceptable performance
        for model in previous_models:
            if (model['accuracy'] is not None and 
                model['accuracy'] >= Config.ACCURACY_ROLLBACK_THRESHOLD):
                
                logger.info(f"Rolling back to model version {model['version']} "
                          f"(accuracy: {model['accuracy']:.3f})")
                
                # Log the rollback event
                self.db.log_deployment_event(
                    model_version=current_model.version,
                    event_type="ROLLBACK",
                    reason=f"{reason}. Rolling back to version {model['version']}",
                    previous_version=model['version']
                )
                
                # In a real implementation, you would trigger the rollback here
                # For example, by calling a deployment service
                # self.deployment_service.activate_model(model['version'])
                
                break
        else:
            logger.warning("No suitable previous model found for rollback")
