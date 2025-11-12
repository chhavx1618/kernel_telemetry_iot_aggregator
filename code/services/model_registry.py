"""Model registry service for managing model versions and deployments."""
import hashlib
import logging
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple

from ..models import ModelMetadata, Database
from ..config import Config

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages model registration, versioning, and deployment."""
    
    def __init__(self, db: Database, registry_dir: Path = Config.MODEL_REGISTRY_DIR):
        """Initialize the model registry.
        
        Args:
            db: Database instance for storing model metadata
            registry_dir: Directory to store model files
        """
        self.db = db
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.file_lock = threading.Lock()
    
    @staticmethod
    def compute_hash(filepath: Path) -> str:
        """Compute SHA-256 hash of a file.
        
        Args:
            filepath: Path to the file to hash
            
        Returns:
            str: Hex digest of the file's SHA-256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def safe_copy_file(self, source_path: Path, dest_path: Path) -> bool:
        """Safely copy a file with error handling and locking.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            
        Returns:
            bool: True if copy was successful, False otherwise
        """
        try:
            with self.file_lock:
                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Use copy2 to preserve metadata
                shutil.copy2(source_path, dest_path)
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to copy file {source_path} to {dest_path}: {e}")
            return False
    
    def register_model(self, source_path: Path, notes: str = "") -> Tuple[Optional[int], bool]:
        """Register a new model in the registry.
        
        Args:
            source_path: Path to the model file to register
            notes: Optional notes about the model
            
        Returns:
            Tuple containing:
                - Model version (int) if successful, None otherwise
                - Boolean indicating if this is a new registration (True) or a duplicate (False)
        """
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source_path}")
            return None, False
            
        # Compute file hash
        file_hash = self.compute_hash(source_path)
        
        # Check if model with this hash already exists
        existing_version = self._get_version_by_hash(file_hash)
        if existing_version is not None:
            logger.info(f"Model with hash {file_hash} already registered as version {existing_version}")
            return existing_version, False
        
        # Register in database
        version = self.db.register_model(
            filename=source_path.name,
            file_hash=file_hash,
            notes=notes
        )
        
        if version is None:
            logger.error("Failed to register model in database")
            return None, False
        
        # Copy model file to registry
        dest_path = self.registry_dir / f"model_v{version}.eim"
        if not self.safe_copy_file(source_path, dest_path):
            logger.error(f"Failed to copy model file to registry for version {version}")
            # TODO: Consider cleaning up the database entry if file copy fails
            return None, False
        
        logger.info(f"Registered new model as version {version}")
        return version, True
    
    def _get_version_by_hash(self, file_hash: str) -> Optional[int]:
        """Check if a model with this hash already exists.
        
        Args:
            file_hash: SHA-256 hash of the model file
            
        Returns:
            int: Model version if found, None otherwise
        """
        # We'll use the database to check for existing hashes
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT version FROM models WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_model_path(self, version: int) -> Optional[Path]:
        """Get the filesystem path for a specific model version.
        
        Args:
            version: Model version number
            
        Returns:
            Path: Path to the model file, or None if not found
        """
        model_path = self.registry_dir / f"model_v{version}.eim"
        return model_path if model_path.exists() else None
    
    def activate_model(self, version: int) -> bool:
        """Activate a specific model version.
        
        Args:
            version: Model version to activate
            
        Returns:
            bool: True if activation was successful, False otherwise
        """
        # Verify model exists
        if not self.get_model_path(version):
            logger.error(f"Cannot activate non-existent model version {version}")
            return False
        
        # Activate model in database
        success = self.db.activate_model(version)
        
        if success:
            logger.info(f"Activated model version {version}")
            self.db.log_deployment_event(
                model_version=version,
                event_type="ACTIVATION",
                reason=f"Manually activated version {version}"
            )
        else:
            logger.error(f"Failed to activate model version {version}")
            
        return success
    
    def get_active_model_path(self) -> Optional[Path]:
        """Get the path to the currently active model.
        
        Returns:
            Path: Path to the active model file, or None if no active model
        """
        active_model = self.db.get_active_model_metadata()
        if not active_model:
            logger.warning("No active model found")
            return None
            
        model_path = self.get_model_path(active_model.version)
        if not model_path or not model_path.exists():
            logger.error(f"Active model file not found: {model_path}")
            return None
            
        return model_path
