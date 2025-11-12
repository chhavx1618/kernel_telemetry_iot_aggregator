"""Service for monitoring and handling model deployments."""
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..models import Database, ModelMetadata
from ..config import Config

logger = logging.getLogger(__name__)

class DeploymentHandler(FileSystemEventHandler):
    """Handles file system events for model deployments."""
    
    def __init__(self, db: Database, registry: Any, engine: Any):
        """Initialize the deployment handler.
        
        Args:
            db: Database instance
            registry: Model registry instance
            engine: Inference engine instance
        """
        self.db = db
        self.registry = registry
        self.engine = engine
        self.processing = set()
        self.lock = threading.Lock()
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in ['.eim', '.tflite', '.h5', '.pt']:
            return
            
        # Skip if already processing this file
        with self.lock:
            if str(file_path) in self.processing:
                return
            self.processing.add(str(file_path))
        
        try:
            logger.info(f"New model file detected: {file_path}")
            
            # Wait for file to be fully written
            if not self._wait_for_file_ready(file_path):
                logger.error(f"File not ready after timeout: {file_path}")
                return
            
            # Register the new model
            version, is_new = self.registry.register_model(
                source_path=file_path,
                notes=f"Auto-deployed from {file_path.name}"
            )
            
            if version is None:
                logger.error(f"Failed to register model: {file_path}")
                return
            
            logger.info(f"Registered model as version {version} (new: {is_new})")
            
            # If this is a new model, activate it
            if is_new:
                logger.info(f"Activating new model version {version}")
                if self.registry.activate_model(version):
                    # Load the new model into the inference engine
                    model_path = self.registry.get_model_path(version)
                    if model_path and self.engine.load_model(version, model_path):
                        logger.info(f"Successfully activated model version {version}")
                        
                        # Log the deployment event
                        self.db.log_deployment_event(
                            model_version=version,
                            event_type="DEPLOYMENT",
                            reason=f"New model deployed from {file_path.name}",
                            previous_version=self.engine.current_version
                        )
                    else:
                        logger.error(f"Failed to load model version {version}")
                else:
                    logger.error(f"Failed to activate model version {version}")
            
            # Move the file to a processed directory to avoid reprocessing
            processed_dir = file_path.parent / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(file_path), str(processed_dir / file_path.name))
                logger.info(f"Moved {file_path.name} to processed directory")
            except Exception as e:
                logger.error(f"Failed to move processed file: {e}")
                
        except Exception as e:
            logger.error(f"Error processing deployment: {e}", exc_info=True)
            
        finally:
            with self.lock:
                self.processing.discard(str(file_path))
    
    def _wait_for_file_ready(self, file_path: Path, timeout: int = 30, check_interval: float = 0.5) -> bool:
        """Wait for a file to be fully written.
        
        Args:
            file_path: Path to the file
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            
        Returns:
            bool: True if file is ready, False if timeout
        """
        start_time = time.time()
        last_size = -1
        
        while (time.time() - start_time) < timeout:
            try:
                current_size = file_path.stat().st_size
                
                # If file size hasn't changed for 2 checks, consider it ready
                if current_size == last_size and current_size > 0:
                    return True
                    
                last_size = current_size
                time.sleep(check_interval)
                
            except (FileNotFoundError, PermissionError):
                # File might be temporarily unavailable
                time.sleep(check_interval)
        
        # Final check
        try:
            return file_path.stat().st_size > 0
        except (FileNotFoundError, PermissionError):
            return False


class DeploymentWatcher:
    """Watches for new model deployments and handles them."""
    
    def __init__(self, db: Database, registry: Any, engine: Any, watch_dir: Path = None):
        """Initialize the deployment watcher.
        
        Args:
            db: Database instance
            registry: Model registry instance
            engine: Inference engine instance
            watch_dir: Directory to watch for new models (defaults to Config.DEPLOYMENT_WATCH_DIR)
        """
        self.db = db
        self.registry = registry
        self.engine = engine
        self.watch_dir = watch_dir or Config.DEPLOYMENT_WATCH_DIR
        self.observer = None
        self.handler = None
    
    def start(self) -> None:
        """Start watching for deployments."""
        if self.observer is not None and self.observer.is_alive():
            logger.warning("Deployment watcher is already running")
            return
        
        # Create watch directory if it doesn't exist
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up the file system observer
        self.handler = DeploymentHandler(self.db, self.registry, self.engine)
        self.observer = Observer()
        self.observer.schedule(
            self.handler,
            str(self.watch_dir),
            recursive=False
        )
        
        # Start the observer
        self.observer.start()
        logger.info(f"Started deployment watcher on {self.watch_dir}")
    
    def stop(self) -> None:
        """Stop watching for deployments."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
            logger.info("Stopped deployment watcher")
    
    def process_existing_files(self) -> None:
        """Process any existing files in the watch directory."""
        if not self.watch_dir.exists():
            return
            
        logger.info(f"Processing existing files in {self.watch_dir}")
        
        for file_path in self.watch_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.eim', '.tflite', '.h5', '.pt']:
                logger.info(f"Processing existing file: {file_path}")
                self.handler.on_created(str(file_path))
