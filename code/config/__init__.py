"""Configuration settings for the MLOps PCB application."""
import os
from pathlib import Path
from typing import Optional

class Config:
    """Application configuration settings."""
    
    # Base directory
    BASE_DIR = Path("/opt/mlops-pcb")
    
    # Directory configurations
    MODEL_REGISTRY_DIR = BASE_DIR / "models"
    DEPLOYMENT_WATCH_DIR = BASE_DIR / "deploy"
    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "output"
    DB_PATH = BASE_DIR / "mlops.db"
    
    # Model configurations
    DEFAULT_MODEL = "model.eim"
    CONFIDENCE_THRESHOLD = 0.5
    
    # Performance monitoring
    DRIFT_DETECTION_WINDOW = 100
    ACCURACY_ROLLBACK_THRESHOLD = 0.7
    PERFORMANCE_DEGRADATION_THRESHOLD = 2.0
    
    # Redis configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8080
    
    # Metrics configuration
    METRICS_PORT = 9090

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        for dir_path in [
            cls.MODEL_REGISTRY_DIR, 
            cls.DEPLOYMENT_WATCH_DIR, 
            cls.UPLOAD_DIR, 
            cls.OUTPUT_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories when the module is imported
Config.ensure_directories()
