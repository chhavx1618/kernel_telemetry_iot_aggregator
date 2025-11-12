"""API routes for the MLOps PCB application."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from ..models import Database, ModelMetadata, InferenceResult, PerformanceMetrics
from ..services.model_registry import ModelRegistry
from ..services.inference_engine import InferenceEngine
from ..services.async_queue import AsyncQueue
from ..config import Config

logger = logging.getLogger(__name__)

# Create a Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize services
db = Database()
registry = ModelRegistry(db)
engine = InferenceEngine(db, registry)
async_queue = AsyncQueue(engine, db)

# Helper functions
def get_services():
    """Get service instances with proper app context."""
    return {
        'db': db,
        'registry': registry,
        'engine': engine,
        'async_queue': async_queue
    }

# Model Management Endpoints

@api_bp.route('/models', methods=['GET'])
def list_models():
    """List all registered models."""
    models = db.get_model_versions()
    return jsonify([asdict(m) for m in models])

@api_bp.route('/models/active', methods=['GET'])
def get_active_model():
    """Get the currently active model."""
    model = db.get_active_model_metadata()
    if not model:
        return jsonify({'error': 'No active model'}), 404
    return jsonify(asdict(model))

@api_bp.route('/models/active', methods=['POST'])
def activate_model():
    """Activate a specific model version."""
    data = request.get_json()
    if not data or 'version' not in data:
        return jsonify({'error': 'Version number is required'}), 400
    
    version = data['version']
    if registry.activate_model(version):
        model_path = registry.get_model_path(version)
        if model_path and engine.load_model(version, model_path):
            return jsonify({'status': 'success', 'active_version': version})
    
    return jsonify({'error': 'Failed to activate model'}), 400

@api_bp.route('/models/upload', methods=['POST'])
def upload_model():
    """Upload a new model file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = Config.UPLOAD_DIR / filename
        
        # Ensure the upload directory exists
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file.save(str(upload_path))
            
            # Register the model
            version, is_new = registry.register_model(
                upload_path,
                notes=request.form.get('notes', '')
            )
            
            if version is None:
                return jsonify({'error': 'Failed to register model'}), 500
                
            return jsonify({
                'status': 'success',
                'version': version,
                'is_new': is_new
            })
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

# Inference Endpoints

@api_bp.route('/inference', methods=['POST'])
def run_inference():
    """Run inference on an image (synchronous)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = Config.UPLOAD_DIR / 'inference' / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file.save(str(upload_path))
            
            # Run inference
            result = engine.predict(upload_path)
            if result is None:
                return jsonify({'error': 'Inference failed'}), 500
            
            # Generate visualization
            output_dir = Config.OUTPUT_DIR / 'visualizations'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{upload_path.stem}_result{upload_path.suffix}"
            engine.visualize_detections(upload_path, result, output_path)
            
            # Prepare response
            response = {
                'model_version': result.model_version,
                'timestamp': result.timestamp,
                'detections': result.detections,
                'inference_time': result.inference_time,
                'visualization_url': f"/api/v1/inference/visualization/{output_path.name}" if output_path.exists() else None
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

@api_bp.route('/inference/async', methods=['POST'])
def run_async_inference():
    """Run inference on an image (asynchronous)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = Config.UPLOAD_DIR / 'inference' / f"{int(datetime.now().timestamp())}_{filename}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file.save(str(upload_path))
            
            # Enqueue the job
            job_id = async_queue.enqueue_inference(
                upload_path,
                callback_url=request.form.get('callback_url'),
                metadata={
                    'client_ip': request.remote_addr,
                    'user_agent': request.user_agent.string,
                    'original_filename': filename
                }
            )
            
            if not job_id:
                return jsonify({'error': 'Failed to enqueue job'}), 500
            
            return jsonify({
                'status': 'queued',
                'job_id': job_id,
                'status_url': f"/api/v1/jobs/{job_id}"
            })
            
        except Exception as e:
            logger.error(f"Error enqueuing job: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

@api_bp.route('/inference/visualization/<filename>')
def get_visualization(filename: str):
    """Get the visualization image for an inference result."""
    try:
        # Sanitize filename to prevent directory traversal
        safe_filename = secure_filename(filename)
        if not safe_filename or safe_filename != filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        file_path = Config.OUTPUT_DIR / 'visualizations' / safe_filename
        if not file_path.exists() or not file_path.is_file():
            return jsonify({'error': 'Visualization not found'}), 404
            
        return send_file(
            str(file_path),
            mimetype='image/jpeg',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Error serving visualization: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Job Management Endpoints

@api_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get the status of an asynchronous job."""
    status = async_queue.get_job_status(job_id)
    if status is None:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(status)

# Metrics and Monitoring Endpoints

@api_bp.route('/metrics')
def get_metrics():
    """Get performance metrics for models."""
    try:
        # Get active model
        active_model = db.get_active_model_metadata()
        if not active_model:
            return jsonify({'error': 'No active model'}), 404
        
        # Get recent metrics
        metrics = db.get_recent_metrics(active_model.version, limit=100)
        
        # Format response
        response = {
            'model_version': active_model.version,
            'metrics': [{
                'timestamp': m.timestamp,
                'accuracy': m.accuracy,
                'avg_confidence': m.avg_confidence,
                'avg_inference_time': m.avg_inference_time,
                'detection_count': m.detection_count
            } for m in metrics]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api_bp.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_ok = db.get_connection() is not None
        
        # Check model is loaded
        model_ok = engine.current_version is not None
        
        # Check Redis connection if using async
        redis_ok = async_queue.redis_client is not None if hasattr(async_queue, 'redis_client') else True
        
        status = {
            'status': 'ok' if all([db_ok, model_ok, redis_ok]) else 'degraded',
            'services': {
                'database': 'ok' if db_ok else 'unavailable',
                'model': 'loaded' if model_ok else 'not loaded',
                'redis': 'connected' if redis_ok else 'disconnected'
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Helper function to convert dataclass to dict
def asdict(obj):
    """Convert a dataclass instance to a dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    elif isinstance(obj, (list, tuple)):
        return [asdict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    else:
        return obj
