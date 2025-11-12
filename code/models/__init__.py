"""Database models and data access layer."""
from pathlib import Path
import sqlite3
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import json

from ..config import Config

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for a machine learning model."""
    version: int
    filename: str
    file_hash: str
    upload_date: str
    deployed_date: Optional[str]
    is_active: bool
    accuracy: float
    avg_inference_time: float
    total_inferences: int
    total_detections: int
    notes: str

@dataclass
class InferenceResult:
    """Result of a model inference."""
    model_version: int
    image_name: str
    timestamp: str
    detections: List[Dict]
    inference_time: float
    confidence_scores: List[float]

@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    model_version: int
    timestamp: str
    accuracy: float
    avg_confidence: float
    avg_inference_time: float
    detection_count: int

class Database:
    """Database access layer for the application."""
    
    def __init__(self, db_path: Path = Config.DB_PATH):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        self.init_db()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection, creating one if it doesn't exist."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def init_db(self) -> None:
        """Initialize the database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                upload_date TEXT NOT NULL,
                deployed_date TEXT,
                is_active INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                avg_inference_time REAL DEFAULT 0.0,
                total_inferences INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                notes TEXT
            )
        ''')
        
        # Create inferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                detections TEXT NOT NULL,
                inference_time REAL NOT NULL,
                confidence_scores TEXT NOT NULL,
                FOREIGN KEY (model_version) REFERENCES models (version)
            )
        ''')
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                avg_confidence REAL,
                avg_inference_time REAL,
                detection_count INTEGER,
                FOREIGN KEY (model_version) REFERENCES models (version)
            )
        ''')
        
        # Create deployment events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deployment_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                reason TEXT,
                previous_version INTEGER,
                FOREIGN KEY (model_version) REFERENCES models (version)
            )
        ''')
        
        conn.commit()
        logger.info("✓ Database initialized")
    
    def register_model(self, filename: str, file_hash: str, notes: str = "") -> int:
        """Register a new model in the database."""
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO models (filename, file_hash, upload_date, notes)
                    VALUES (?, ?, ?, ?)
                ''', (filename, file_hash, datetime.now().isoformat(), notes))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                cursor.execute('SELECT version FROM models WHERE file_hash = ?', (file_hash,))
                return cursor.fetchone()[0]
    
    def activate_model(self, version: int) -> bool:
        """Activate a model version."""
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # First, deactivate all models
            cursor.execute('UPDATE models SET is_active = 0')
            
            # Then activate the specified model
            cursor.execute('''
                UPDATE models 
                SET is_active = 1, 
                    deployed_date = ? 
                WHERE version = ?
            ''', (datetime.now().isoformat(), version))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_model_metadata(self, version: int) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version."""
        cursor = self.get_connection().cursor()
        cursor.execute('SELECT * FROM models WHERE version = ?', (version,))
        row = cursor.fetchone()
        
        if row:
            return ModelMetadata(
                version=row['version'],
                filename=row['filename'],
                file_hash=row['file_hash'],
                upload_date=row['upload_date'],
                deployed_date=row['deployed_date'],
                is_active=bool(row['is_active']),
                accuracy=row['accuracy'],
                avg_inference_time=row['avg_inference_time'],
                total_inferences=row['total_inferences'],
                total_detections=row['total_detections'],
                notes=row['notes']
            )
        return None
    
    def get_active_model_metadata(self) -> Optional[ModelMetadata]:
        """Get metadata for the currently active model."""
        cursor = self.get_connection().cursor()
        cursor.execute('SELECT * FROM models WHERE is_active = 1')
        row = cursor.fetchone()
        
        if row:
            return self.get_model_metadata(row['version'])
        return None
    
    def log_inference(self, inference: InferenceResult) -> int:
        """Log an inference result to the database."""
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inferences 
                (model_version, image_name, timestamp, detections, inference_time, confidence_scores)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                inference.model_version,
                inference.image_name,
                inference.timestamp,
                json.dumps(inference.detections),
                inference.inference_time,
                json.dumps(inference.confidence_scores)
            ))
            
            # Update model statistics
            cursor.execute('''
                UPDATE models 
                SET total_inferences = total_inferences + 1,
                    total_detections = total_detections + ?
                WHERE version = ?
            ''', (len(inference.detections), inference.model_version))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_metrics(self, metrics: PerformanceMetrics) -> int:
        """Log performance metrics for a model."""
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics 
                (model_version, timestamp, accuracy, avg_confidence, avg_inference_time, detection_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metrics.model_version,
                metrics.timestamp,
                metrics.accuracy,
                metrics.avg_confidence,
                metrics.avg_inference_time,
                metrics.detection_count
            ))
            
            # Update model's accuracy and inference time
            cursor.execute('''
                UPDATE models 
                SET accuracy = ?,
                    avg_inference_time = ?
                WHERE version = ?
            ''', (metrics.accuracy, metrics.avg_inference_time, metrics.model_version))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_deployment_event(self, model_version: int, event_type: str, reason: str = None, 
                           previous_version: int = None) -> int:
        """Log a deployment event."""
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO deployment_events 
                (model_version, event_type, timestamp, reason, previous_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                model_version,
                event_type,
                datetime.now().isoformat(),
                reason,
                previous_version
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_metrics(self, model_version: int, limit: int = 100) -> List[PerformanceMetrics]:
        """Get recent performance metrics for a model."""
        cursor = self.get_connection().cursor()
        cursor.execute('''
            SELECT * FROM metrics 
            WHERE model_version = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (model_version, limit))
        
        return [
            PerformanceMetrics(
                model_version=row['model_version'],
                timestamp=row['timestamp'],
                accuracy=row['accuracy'],
                avg_confidence=row['avg_confidence'],
                avg_inference_time=row['avg_inference_time'],
                detection_count=row['detection_count']
            )
            for row in cursor.fetchall()
        ]
    
    def get_model_versions(self) -> List[ModelMetadata]:
        """Get all registered model versions."""
        cursor = self.get_connection().cursor()
        cursor.execute('SELECT version FROM models ORDER BY version DESC')
        
        return [
            self.get_model_metadata(row['version'])
            for row in cursor.fetchall()
        ]
