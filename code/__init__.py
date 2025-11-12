"""MLOps PCB - Main application package."""
from pathlib import Path

# Initialize configuration
from .config import Config

# Ensure required directories exist
Config.ensure_directories()

# Import services after config is set up
from .models import Database
from .services.model_registry import ModelRegistry
from .services.inference_engine import InferenceEngine
from .services.performance_monitor import PerformanceMonitor
from .services.deployment_watcher import DeploymentWatcher
from .services.async_queue import AsyncQueue

# Initialize core services
db = Database()
registry = ModelRegistry(db)
engine = InferenceEngine(db, registry)
async_queue = AsyncQueue(engine, db)
performance_monitor = PerformanceMonitor(db)
deployment_watcher = DeploymentWatcher(db, registry, engine)

# Import API routes after services are initialized
from .api.routes import api_bp

# Create Flask application
def create_app():
    """Create and configure the Flask application."""
    from flask import Flask
    
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Add health check endpoint at root
    @app.route('/')
    def index():
        return {
            'name': 'MLOps PCB API',
            'status': 'running',
            'endpoints': {
                'api_docs': '/api/v1',
                'health': '/api/v1/health',
                'models': '/api/v1/models',
                'inference': '/api/v1/inference',
                'metrics': '/api/v1/metrics'
            }
        }
    
    return app

def start_services():
    """Start background services."""
    # Start performance monitoring
    performance_monitor.start()
    
    # Start deployment watcher
    deployment_watcher.start()
    
    # Process any existing files in the watch directory
    deployment_watcher.process_existing_files()

# Run the application if executed directly
if __name__ == '__main__':
    import argparse
    import logging
    from waitress import serve
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='MLOps PCB Application')
    parser.add_argument('--host', default=Config.API_HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=Config.API_PORT, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and configure the app
    app = create_app()
    
    # Start background services
    start_services()
    
    # Run the application
    if args.debug:
        # Use Flask's development server in debug mode
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # Use Waitress in production
        print(f"Starting server on {args.host}:{args.port}")
        serve(app, host=args.host, port=args.port)
