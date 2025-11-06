#!/usr/bin/env python3

import os
import sys
import json
import time
import sqlite3
import hashlib
import threading
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename

import redis

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

import cv2
from edge_impulse_linux.image import ImageImpulseRunner

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Config:
    BASE_DIR = Path("/opt/mlops-pcb")
    MODEL_REGISTRY_DIR = BASE_DIR / "models"
    DEPLOYMENT_WATCH_DIR = BASE_DIR / "deploy"
    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "output"
    DB_PATH = BASE_DIR / "mlops.db"
    
    DEFAULT_MODEL = "model.eim"
    CONFIDENCE_THRESHOLD = 0.5
    
    DRIFT_DETECTION_WINDOW = 100
    ACCURACY_ROLLBACK_THRESHOLD = 0.7
    PERFORMANCE_DEGRADATION_THRESHOLD = 2.0
    
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    API_HOST = "0.0.0.0"
    API_PORT = 8080
    
    METRICS_PORT = 9090

    @classmethod
    def ensure_directories(cls):
        for dir_path in [cls.MODEL_REGISTRY_DIR, cls.DEPLOYMENT_WATCH_DIR, 
                         cls.UPLOAD_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)

inference_counter = Counter('pcb_inferences_total', 'Total inference requests', ['model_version', 'status'])
detection_counter = Counter('pcb_detections_total', 'Total defects detected', ['model_version', 'label'])
deployment_counter = Counter('pcb_model_deployments_total', 'Model deployment events', ['status'])
rollback_counter = Counter('pcb_model_rollbacks_total', 'Model rollback events')

inference_latency = Histogram('pcb_inference_latency_seconds', 'Inference latency', ['model_version'])
confidence_histogram = Histogram('pcb_confidence_score', 'Detection confidence scores', 
                                 ['model_version'], buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

active_model_version = Gauge('pcb_active_model_version', 'Currently active model version')
model_accuracy = Gauge('pcb_model_accuracy', 'Model accuracy (last N predictions)', ['model_version'])
drift_score = Gauge('pcb_drift_score', 'Data drift score', ['model_version'])
inference_queue_size = Gauge('pcb_inference_queue_size', 'Redis queue size')

@dataclass
class ModelMetadata:
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
    model_version: int
    image_name: str
    timestamp: str
    detections: List[Dict]
    inference_time: float
    confidence_scores: List[float]

@dataclass
class PerformanceMetrics:
    model_version: int
    timestamp: str
    accuracy: float
    avg_confidence: float
    avg_inference_time: float
    detection_count: int

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        self.init_db()
    
    def get_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
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
        logging.info("✓ Database initialized")
    
    def register_model(self, filename: str, file_hash: str, notes: str = "") -> int:
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
    
    def activate_model(self, version: int):
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('UPDATE models SET is_active = 0')
            
            cursor.execute('''
                UPDATE models 
                SET is_active = 1, deployed_date = ? 
                WHERE version = ?
            ''', (datetime.now().isoformat(), version))
            
            conn.commit()
    
    def get_active_model(self) -> Optional[ModelMetadata]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models WHERE is_active = 1 LIMIT 1')
        row = cursor.fetchone()
        
        if row:
            return ModelMetadata(**dict(row))
        return None
    
    def get_model_by_version(self, version: int) -> Optional[ModelMetadata]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models WHERE version = ?', (version,))
        row = cursor.fetchone()
        
        if row:
            return ModelMetadata(**dict(row))
        return None
    
    def list_models(self) -> List[ModelMetadata]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models ORDER BY version DESC')
        rows = cursor.fetchall()
        return [ModelMetadata(**dict(row)) for row in rows]
    
    def log_inference(self, result: InferenceResult):
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inferences 
                (model_version, image_name, timestamp, detections, inference_time, confidence_scores)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                result.model_version,
                result.image_name,
                result.timestamp,
                json.dumps(result.detections),
                result.inference_time,
                json.dumps(result.confidence_scores)
            ))
            
            cursor.execute('''
                UPDATE models 
                SET total_inferences = total_inferences + 1,
                    total_detections = total_detections + ?
                WHERE version = ?
            ''', (len(result.detections), result.model_version))
            
            conn.commit()
    
    def log_metrics(self, metrics: PerformanceMetrics):
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
            
            conn.commit()
    
    def log_deployment_event(self, version: int, event_type: str, reason: str, prev_version: Optional[int] = None):
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO deployment_events 
                (model_version, event_type, timestamp, reason, previous_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (version, event_type, datetime.now().isoformat(), reason, prev_version))
            
            conn.commit()
    
    def get_recent_inferences(self, model_version: int, limit: int = 100) -> List[InferenceResult]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_version, image_name, timestamp, detections, inference_time, confidence_scores
            FROM inferences 
            WHERE model_version = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (model_version, limit))
        
        rows = cursor.fetchall()
        results = []
        for row in rows:
            data = dict(row)
            data['detections'] = json.loads(data['detections'])
            data['confidence_scores'] = json.loads(data['confidence_scores'])
            results.append(InferenceResult(**data))
        
        return results

class ModelRegistry:
    def __init__(self, db: Database, registry_dir: Path):
        self.db = db
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.file_lock = threading.Lock()
    
    def compute_hash(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def safe_copy_file(self, source_path: Path, dest_path: Path) -> bool:
        try:
            with open(source_path, 'rb') as src:
                data = src.read()
            
            temp_path = dest_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as dst:
                dst.write(data)
            
            # Make the file executable
            import stat
            os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | 
                               stat.S_IRGRP | stat.S_IXGRP | 
                               stat.S_IROTH | stat.S_IXOTH)
            
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            
            return True
        except Exception as e:
            logging.error(f"Failed to copy file: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            return False
    
    def register_model(self, source_path: Path, notes: str = "") -> int:
        with self.file_lock:
            file_hash = self.compute_hash(source_path)
            filename = source_path.name
            
            # Check if this exact model already exists
            existing_version = self._get_version_by_hash(file_hash)
            if existing_version:
                logging.info(f"Model with same hash already exists as v{existing_version}, skipping registration")
                return existing_version
            
            version = self.db.register_model(filename, file_hash, notes)
            
            dest_path = self.registry_dir / f"model_v{version}.eim"
            
            if not self.safe_copy_file(source_path, dest_path):
                raise RuntimeError(f"Failed to copy model file to registry")
            
            logging.info(f"✓ Model registered: v{version} ({filename})")
            return version
    
    def _get_version_by_hash(self, file_hash: str) -> Optional[int]:
        """Check if a model with this hash already exists"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT version FROM models WHERE file_hash = ?', (file_hash,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_model_path(self, version: int) -> Path:
        return self.registry_dir / f"model_v{version}.eim"
    
    def activate_model(self, version: int) -> bool:
        model_path = self.get_model_path(version)
        
        if not model_path.exists():
            logging.error(f"Model v{version} not found in registry")
            return False
        
        self.db.activate_model(version)
        self.db.log_deployment_event(version, "ACTIVATE", "Manual activation")
        
        active_model_version.set(version)
        deployment_counter.labels(status='success').inc()
        
        logging.info(f"✓ Model v{version} activated")
        return True
    
    def get_active_model_path(self) -> Optional[Path]:
        active = self.db.get_active_model()
        if active:
            return self.get_model_path(active.version)
        return None

class InferenceEngine:
    def __init__(self, db: Database, registry: ModelRegistry):
        self.db = db
        self.registry = registry
        self.runner = None
        self.current_version = None
        self.model_info = None
        self.lock = threading.Lock()
        
        self.load_active_model()
    
    def load_active_model(self):
        active = self.db.get_active_model()
        
        if not active:
            logging.warning("⚠️ No active model found in database")
            return False
        
        model_path = self.registry.get_model_path(active.version)
        
        if not model_path.exists():
            logging.error(f"❌ Active model v{active.version} not found at {model_path}")
            return False
        
        logging.info(f"📥 Loading active model v{active.version} from {model_path}")
        success = self.load_model(active.version, model_path)
        
        if success:
            logging.info(f"✅ Active model v{active.version} loaded successfully")
        else:
            logging.error(f"❌ Failed to load active model v{active.version}")
        
        return success
    
    def load_model(self, version: int, model_path: Path) -> bool:
        with self.lock:
            try:
                # Stop existing runner if any
                if self.runner:
                    try:
                        self.runner.stop()
                    except:
                        pass
                    self.runner = None
                    self.model_info = None
                
                logging.info(f"🔄 Loading model v{version} from {model_path}")
                
                # Verify file exists and is accessible
                if not model_path.exists():
                    logging.error(f"❌ Model file does not exist: {model_path}")
                    return False
                
                # Check file permissions
                if not os.access(model_path, os.X_OK):
                    logging.warning(f"⚠️ Model file not executable, setting permissions...")
                    import stat
                    os.chmod(model_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | 
                                        stat.S_IRGRP | stat.S_IXGRP | 
                                        stat.S_IROTH | stat.S_IXOTH)
                
                # Initialize the runner
                self.runner = ImageImpulseRunner(str(model_path))
                self.model_info = self.runner.init()
                
                if not self.model_info:
                    logging.error(f"❌ Model init returned None")
                    self.runner = None
                    return False
                
                self.current_version = version
                
                project_name = self.model_info.get('project', {}).get('name', 'Unknown')
                logging.info(f"✅ Model v{version} loaded successfully: {project_name}")
                
                return True
                
            except Exception as e:
                logging.error(f"❌ Failed to load model v{version}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                self.runner = None
                self.model_info = None
                self.current_version = None
                return False
    
    def predict(self, image_path: Path, confidence_threshold: float = Config.CONFIDENCE_THRESHOLD) -> InferenceResult:
        if not self.runner:
            logging.error("❌ No model loaded in predict()")
            raise RuntimeError("No model loaded")
        
        if not self.model_info:
            logging.error("❌ Model info not available")
            raise RuntimeError("Model not properly initialized")
        
        start_time = time.time()
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        input_width = self.model_info['model_parameters']['image_input_width']
        input_height = self.model_info['model_parameters']['image_input_height']
        img_resized = cv2.resize(img, (input_width, input_height))
        
        with self.lock:
            features, _ = self.runner.get_features_from_image(img_resized)
            res = self.runner.classify(features)
        
        inference_time = time.time() - start_time
        
        detections = []
        confidence_scores = []
        
        if 'bounding_boxes' in res['result']:
            for bb in res['result']['bounding_boxes']:
                if bb['value'] >= confidence_threshold:
                    detections.append({
                        'label': bb['label'],
                        'confidence': bb['value'],
                        'bbox': [bb['x'], bb['y'], bb['width'], bb['height']]
                    })
                    confidence_scores.append(bb['value'])
                    
                    detection_counter.labels(
                        model_version=f"v{self.current_version}",
                        label=bb['label']
                    ).inc()
                    
                    confidence_histogram.labels(
                        model_version=f"v{self.current_version}"
                    ).observe(bb['value'])
        
        result = InferenceResult(
            model_version=self.current_version,
            image_name=image_path.name,
            timestamp=datetime.now().isoformat(),
            detections=detections,
            inference_time=inference_time,
            confidence_scores=confidence_scores
        )
        
        inference_counter.labels(
            model_version=f"v{self.current_version}",
            status='success'
        ).inc()
        
        inference_latency.labels(
            model_version=f"v{self.current_version}"
        ).observe(inference_time)
        
        self.db.log_inference(result)
        
        return result
    
    def visualize_detections(self, image_path: Path, result: InferenceResult, output_path: Path):
        img = cv2.imread(str(image_path))
        
        for det in result.detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            label = f"{det['label']} {det['confidence']:.2%}"
            cv2.putText(img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), img)

class PerformanceMonitor:
    def __init__(self, db: Database, engine: InferenceEngine, registry: ModelRegistry):
        self.db = db
        self.engine = engine
        self.registry = registry
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logging.info("✓ Performance monitor started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        while self.running:
            try:
                self._check_performance()
                self._detect_drift()
                time.sleep(60)
            except Exception as e:
                logging.error(f"Monitor error: {e}")
    
    def _check_performance(self):
        if not self.engine.current_version:
            return
        
        recent = self.db.get_recent_inferences(
            self.engine.current_version, 
            Config.DRIFT_DETECTION_WINDOW
        )
        
        if len(recent) < 10:
            return
        
        total_detections = sum(len(r.detections) for r in recent)
        avg_inference_time = sum(r.inference_time for r in recent) / len(recent)
        
        all_confidences = []
        for r in recent:
            all_confidences.extend(r.confidence_scores)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        metrics = PerformanceMetrics(
            model_version=self.engine.current_version,
            timestamp=datetime.now().isoformat(),
            accuracy=avg_confidence,
            avg_confidence=avg_confidence,
            avg_inference_time=avg_inference_time,
            detection_count=total_detections
        )
        
        self.db.log_metrics(metrics)
        
        model_accuracy.labels(
            model_version=f"v{self.engine.current_version}"
        ).set(avg_confidence)
        
        if avg_confidence < Config.ACCURACY_ROLLBACK_THRESHOLD:
            logging.warning(f"⚠️ Model accuracy dropped to {avg_confidence:.2%}")
            self._trigger_rollback("Low accuracy")
    
    def _detect_drift(self):
        if not self.engine.current_version:
            return
        
        recent = self.db.get_recent_inferences(self.engine.current_version, 50)
        if len(recent) < 50:
            return
        
        all_confidences = []
        for r in recent:
            all_confidences.extend(r.confidence_scores)
        
        if not all_confidences:
            return
        
        mean_conf = sum(all_confidences) / len(all_confidences)
        variance = sum((x - mean_conf) ** 2 for x in all_confidences) / len(all_confidences)
        drift_value = variance
        
        drift_score.labels(model_version=f"v{self.engine.current_version}").set(drift_value)
        
        if drift_value > 0.15:
            logging.warning(f"⚠️ Data drift detected: {drift_value:.3f}")
    
    def _trigger_rollback(self, reason: str):
        current_version = self.engine.current_version
        
        models = self.db.list_models()
        previous = None
        for m in models:
            if m.version < current_version and m.deployed_date:
                previous = m
                break
        
        if not previous:
            logging.error("No previous model to rollback to")
            return
        
        logging.warning(f"🔄 Rolling back from v{current_version} to v{previous.version}")
        
        self.registry.activate_model(previous.version)
        
        self.db.log_deployment_event(
            previous.version,
            "ROLLBACK",
            reason,
            current_version
        )
        
        rollback_counter.inc()
        
        self.engine.load_active_model()

class DeploymentWatcher(FileSystemEventHandler):
    def __init__(self, db: Database, registry: ModelRegistry, engine: InferenceEngine):
        self.db = db
        self.registry = registry
        self.engine = engine
        self.processing = set()
        self.lock = threading.Lock()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        
        if filepath.suffix != '.eim':
            logging.info(f"Ignoring non-.eim file: {filepath.name}")
            return
        
        with self.lock:
            if str(filepath) in self.processing:
                return
            self.processing.add(str(filepath))
        
        try:
            self._wait_for_file_ready(filepath)
            
            logging.info(f"📦 New model detected: {filepath.name}")
            
            # Register the model (will return existing version if same hash)
            version = self.registry.register_model(filepath, "Auto-deployed")
            
            # Only activate if it's a new model or we want to force activation
            current_active = self.db.get_active_model()
            if not current_active or current_active.version != version:
                logging.info(f"Activating model v{version}")
                self.registry.activate_model(version)
                self.engine.load_active_model()
            else:
                logging.info(f"Model v{version} already active, skipping activation")
            
            # Move to processed folder
            processed_dir = filepath.parent / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            processed_path = processed_dir / f"{timestamp}_{filepath.name}"
            if self.registry.safe_copy_file(filepath, processed_path):
                filepath.unlink()
            
            logging.info(f"✓ Model v{version} deployment complete")
            
        except Exception as e:
            logging.error(f"Failed to deploy {filepath.name}: {e}")
            deployment_counter.labels(status='failed').inc()
        finally:
            with self.lock:
                self.processing.discard(str(filepath))
    
    def _wait_for_file_ready(self, filepath: Path, timeout: int = 10):
        start_time = time.time()
        last_size = -1
        
        while time.time() - start_time < timeout:
            try:
                current_size = filepath.stat().st_size
                if current_size == last_size and current_size > 0:
                    time.sleep(1)
                    if filepath.stat().st_size == current_size:
                        return
                last_size = current_size
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)
        
        if not filepath.exists() or filepath.stat().st_size == 0:
            raise RuntimeError(f"File {filepath.name} not ready after {timeout}s")

def start_deployment_watcher(db: Database, registry: ModelRegistry, engine: InferenceEngine):
    event_handler = DeploymentWatcher(db, registry, engine)
    observer = Observer()
    observer.schedule(event_handler, str(Config.DEPLOYMENT_WATCH_DIR), recursive=False)
    observer.start()
    logging.info(f"👀 Watching for models in: {Config.DEPLOYMENT_WATCH_DIR}")
    return observer

class AsyncQueue:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.redis_client = None
        self.enabled = False
        self.running = False
        self.worker_thread = None
        
        self._connect_redis()
    
    def _connect_redis(self):
        try:
            logging.info(f"Attempting to connect to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=False,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError],
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logging.info(f"✓ Redis connected successfully at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except redis.ConnectionError as e:
            logging.warning(f"⚠️ Redis connection failed: {e}")
            logging.warning(f"   Make sure Redis is running: sudo systemctl status redis-server")
            logging.warning(f"   Or start it with: sudo systemctl start redis-server")
            self.enabled = False
            self.redis_client = None
        except Exception as e:
            logging.warning(f"⚠️ Redis not available: {e}, async queue disabled")
            self.enabled = False
            self.redis_client = None
    
    def start_worker(self):
        if not self.enabled:
            logging.warning("Cannot start worker: Redis not available")
            return
        
        # Verify model is loaded before starting worker
        if not self.engine.runner:
            logging.warning("⚠️ No model loaded, attempting to load active model...")
            if not self.engine.load_active_model():
                logging.error("❌ Cannot start async worker: No model available")
                return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logging.info("✓ Async worker started")
    
    def stop_worker(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _worker_loop(self):
        consecutive_errors = 0
        max_errors = 5
        
        logging.info("🔄 Async worker loop started, waiting for jobs...")
        
        while self.running:
            try:
                # Non-blocking check with 1 second timeout
                job_data = self.redis_client.blpop('inference_queue', timeout=1)
                
                if job_data:
                    _, job_json = job_data
                    job = json.loads(job_json)
                    logging.info(f"📥 Processing async job: {job.get('image_path', 'unknown')}")
                    
                    self._process_job(job)
                    consecutive_errors = 0
                    logging.info(f"✅ Job completed successfully")
                
                # Update queue size metric
                try:
                    queue_size = self.redis_client.llen('inference_queue')
                    inference_queue_size.set(queue_size)
                except:
                    pass
                
            except redis.ConnectionError as e:
                consecutive_errors += 1
                logging.error(f"❌ Redis connection error in worker (attempt {consecutive_errors}/{max_errors}): {e}")
                
                if consecutive_errors >= max_errors:
                    logging.error("🛑 Too many consecutive Redis errors, stopping worker")
                    self.enabled = False
                    break
                
                time.sleep(5)
                logging.info("🔄 Attempting to reconnect to Redis...")
                self._connect_redis()
                
            except json.JSONDecodeError as e:
                logging.error(f"❌ Invalid JSON in job: {e}")
                consecutive_errors = 0
                
            except Exception as e:
                logging.error(f"❌ Worker error: {e}")
                import traceback
                logging.error(traceback.format_exc())
                time.sleep(1)
    
    def _process_job(self, job: dict):
        try:
            image_path = Path(job['image_path'])
            output_path = Path(job['output_path'])
            
            if not image_path.exists():
                logging.error(f"❌ Image not found: {image_path}")
                return
            
            # Check if model is loaded
            if not self.engine.runner:
                logging.error(f"❌ No model loaded, cannot process job")
                # Try to reload the active model
                if not self.engine.load_active_model():
                    logging.error(f"❌ Failed to reload model, skipping job")
                    return
            
            result = self.engine.predict(image_path)
            
            self.engine.visualize_detections(image_path, result, output_path)
            
            logging.info(f"✓ Async job completed: {image_path.name}")
            
        except Exception as e:
            logging.error(f"❌ Job failed for {job.get('image_path', 'unknown')}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def enqueue(self, image_path: Path, output_path: Path) -> bool:
        if not self.enabled or not self.redis_client:
            logging.warning("⚠️ Cannot enqueue: Redis not available")
            return False
        
        try:
            job = {
                'image_path': str(image_path),
                'output_path': str(output_path),
                'timestamp': datetime.now().isoformat()
            }
            
            job_json = json.dumps(job)
            self.redis_client.rpush('inference_queue', job_json)
            
            queue_size = self.redis_client.llen('inference_queue')
            logging.info(f"✓ Job enqueued: {image_path.name} (queue size: {queue_size})")
            return True
            
        except redis.ConnectionError as e:
            logging.error(f"❌ Redis connection error during enqueue: {e}")
            self.enabled = False
            return False
        except Exception as e:
            logging.error(f"❌ Failed to enqueue job: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def get_queue_size(self) -> int:
        if not self.enabled:
            return 0
        try:
            return self.redis_client.llen('inference_queue')
        except:
            return 0

def create_app(db: Database, registry: ModelRegistry, engine: InferenceEngine, async_queue: AsyncQueue):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    
    DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>MLOps PCB Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { opacity: 0.9; font-size: 1.1em; }
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
        }
        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }
        .tab:hover { background: #e9ecef; }
        .tab.active {
            background: white;
            border-bottom-color: #667eea;
            color: #667eea;
        }
        .tab-content {
            display: none;
            padding: 30px;
        }
        .tab-content.active { display: block; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
        }
        .model-list {
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
        }
        .model-item {
            background: white;
            margin: 10px;
            padding: 20px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .model-item.active {
            border: 3px solid #28a745;
            background: #d4edda;
        }
        .model-info { flex: 1; }
        .model-version {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        .model-meta {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            margin: 5px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        button.danger {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        }
        button.success {
            background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 50px;
            text-align: center;
            background: #f8f9fa;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #e9ecef;
            border-color: #764ba2;
        }
        input[type="file"] { display: none; }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
        }
        .error {
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        .badge.active {
            background: #28a745;
            color: white;
        }
        .badge.inactive {
            background: #6c757d;
            color: white;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        .alert {
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }
        .alert.warning {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 MLOps PCB Detection System</h1>
            <p class="subtitle">Production-grade Edge Impulse deployment on Raspberry Pi 4</p>
        </header>
        <div class="tabs">
            <div class="tab active" onclick="switchTab('dashboard')">📊 Dashboard</div>
            <div class="tab" onclick="switchTab('models')">🤖 Models</div>
            <div class="tab" onclick="switchTab('inference')">🎯 Inference</div>
            <div class="tab" onclick="switchTab('deploy')">🚀 Deploy</div>
            <div class="tab" onclick="switchTab('metrics')">📈 Metrics</div>
        </div>
        <div id="dashboard" class="tab-content active">
            <h2>System Overview</h2>
            <div class="stats-grid" id="systemStats">
                <div class="stat-card">
                    <div class="stat-label">Active Model</div>
                    <div class="stat-value" id="activeVersion">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Inferences</div>
                    <div class="stat-value" id="totalInferences">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Detections</div>
                    <div class="stat-value" id="totalDetections">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Confidence</div>
                    <div class="stat-value" id="avgConfidence">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Redis Queue</div>
                    <div class="stat-value" id="queueSize">0</div>
                </div>
            </div>
        </div>
        <div id="models" class="tab-content">
            <h2>Model Registry</h2>
            <button onclick="loadModels()" class="success">🔄 Refresh</button>
            <div id="modelList" class="model-list"></div>
        </div>
        <div id="inference" class="tab-content">
            <h2>Run Inference</h2>
            <div class="upload-area" onclick="document.getElementById('inferenceInput').click()">
                <h3>📁 Click to browse</h3>
                <p>JPG, PNG, BMP</p>
                <input type="file" id="inferenceInput" accept="image/*">
            </div>
            <button onclick="runInference(false)" class="success">▶️ Run Inference</button>
            <div id="inferenceResult"></div>
        </div>
        <div id="deploy" class="tab-content">
            <h2>Deploy New Model</h2>
            <p><strong>Auto:</strong> <code>{{ deploy_dir }}</code></p>
            <div class="upload-area" onclick="document.getElementById('deployInput').click()">
                <h3>📦 Upload .eim</h3>
                <input type="file" id="deployInput" accept=".eim">
            </div>
            <label>
                Notes: <input type="text" id="deployNotes" placeholder="Optional" style="width: 400px; padding: 8px;">
            </label>
            <button onclick="deployModel()" class="success">🚀 Deploy</button>
            <div id="deployResult"></div>
        </div>
        <div id="metrics" class="tab-content">
            <h2>Prometheus Metrics</h2>
            <p>Endpoint: <code>http://{{ host }}:{{ api_port }}/metrics</code></p>
            <button onclick="window.open('/metrics', '_blank')" class="success">📊 View Metrics</button>
        </div>
    </div>
    <script>
        let redisAvailable = false;
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            if (tabName === 'models') loadModels();
            if (tabName === 'dashboard') loadDashboard();
        }
        async function loadDashboard() {
            try {
                const [statsRes, healthRes] = await Promise.all([
                    fetch('/api/stats'),
                    fetch('/health')
                ]);
                const stats = await statsRes.json();
                const health = await healthRes.json();
                document.getElementById('activeVersion').textContent = 'v' + stats.active_version;
                document.getElementById('totalInferences').textContent = stats.total_inferences;
                document.getElementById('totalDetections').textContent = stats.total_detections;
                document.getElementById('avgConfidence').textContent = (stats.avg_confidence * 100).toFixed(1) + '%';
                document.getElementById('queueSize').textContent = stats.queue_size || 0;
                redisAvailable = health.redis_available;
            } catch (e) {
                console.error('Failed to load dashboard:', e);
            }
        }
        async function loadModels() {
            const res = await fetch('/api/models');
            const models = await res.json();
            let html = '';
            models.forEach(m => {
                html += `
                    <div class="model-item ${m.is_active ? 'active' : ''}">
                        <div class="model-info">
                            <div class="model-version">
                                Model v${m.version}
                                ${m.is_active ? '<span class="badge active">ACTIVE</span>' : '<span class="badge inactive">INACTIVE</span>'}
                            </div>
                            <div class="model-meta">
                                ${m.filename} | ${new Date(m.upload_date).toLocaleString()} | Inf: ${m.total_inferences} | Det: ${m.total_detections}
                            </div>
                            ${m.notes ? '<div class="model-meta">' + m.notes + '</div>' : ''}
                        </div>
                        <div>
                            ${!m.is_active ? '<button onclick="activateModel(' + m.version + ')" class="success">✓ Activate</button>' : ''}
                            <button onclick="deleteModel(' + m.version + ')" class="danger">🗑️ Delete</button>
                        </div>
                    </div>
                `;
            });
            document.getElementById('modelList').innerHTML = html || '<p style="padding:20px;">No models</p>';
        }
        async function activateModel(version) {
            if (!confirm(`Activate model v${version}?`)) return;
            const res = await fetch('/api/models/' + version + '/activate', { method: 'POST' });
            const data = await res.json();
            alert(data.message || 'Model activated!');
            loadModels();
            loadDashboard();
        }
        async function deleteModel(version) {
            if (!confirm(`Delete model v${version}?`)) return;
            const res = await fetch('/api/models/' + version, { method: 'DELETE' });
            const data = await res.json();
            alert(data.message || 'Model deleted!');
            loadModels();
        }
        async function runInference(async) {
            const input = document.getElementById('inferenceInput');
            if (!input.files[0]) {
                alert('Please select an image');
                return;
            }
            if (async && !redisAvailable) {
                alert('Redis not available');
                return;
            }
            const formData = new FormData();
            formData.append('image', input.files[0]);
            formData.append('async', async);
            const resultDiv = document.getElementById('inferenceResult');
            resultDiv.innerHTML = '<p>Processing...</p>';
            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                let html = '<div class="result">';
                if (async) {
                    html += '<p><strong>✓ Job queued</strong></p>';
                } else {
                    html += `<p><strong>Model:</strong> v${data.model_version}</p>`;
                    html += `<p><strong>Time:</strong> ${data.inference_time.toFixed(3)}s</p>`;
                    html += `<p><strong>Detections:</strong> ${data.detections.length}</p>`;
                    data.detections.forEach((d, i) => {
                        html += `<p>${i+1}. ${d.label} (${(d.confidence*100).toFixed(1)}%)</p>`;
                    });
                    if (data.image_url) {
                        html += `<img src="${data.image_url}?t=${Date.now()}" alt="Result">`;
                    }
                }
                html += '</div>';
                resultDiv.innerHTML = html;
                if (async) {
                    setTimeout(loadDashboard, 500);
                }
            } catch (e) {
                resultDiv.innerHTML = '<div class="result error"><p>Error: ' + e.message + '</p></div>';
            }
        }
        async function deployModel() {
            const input = document.getElementById('deployInput');
            if (!input.files[0]) {
                alert('Please select a .eim file');
                return;
            }
            const formData = new FormData();
            formData.append('model', input.files[0]);
            formData.append('notes', document.getElementById('deployNotes').value);
            const resultDiv = document.getElementById('deployResult');
            resultDiv.innerHTML = '<p>Deploying...</p>';
            try {
                const res = await fetch('/api/models/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (res.ok) {
                    resultDiv.innerHTML = `<div class="result"><p><strong>✓ Model v${data.version} deployed!</strong></p></div>`;
                    document.getElementById('deployNotes').value = '';
                    input.value = '';
                } else {
                    resultDiv.innerHTML = `<div class="result error"><p>Error: ${data.error}</p></div>`;
                }
            } catch (e) {
                resultDiv.innerHTML = '<div class="result error"><p>Error: ' + e.message + '</p></div>';
            }
        }
        window.addEventListener('load', loadDashboard);
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
    '''
    
    @app.route('/')
    def dashboard():
        return render_template_string(
            DASHBOARD_HTML,
            deploy_dir=Config.DEPLOYMENT_WATCH_DIR,
            host=Config.API_HOST if Config.API_HOST != '0.0.0.0' else 'localhost',
            api_port=Config.API_PORT
        )
    
    @app.route('/api/stats')
    def get_stats():
        active = db.get_active_model()
        stats = {
            'active_version': active.version if active else 0,
            'total_inferences': active.total_inferences if active else 0,
            'total_detections': active.total_detections if active else 0,
            'avg_confidence': active.accuracy if active else 0.0,
            'queue_size': async_queue.get_queue_size()
        }
        return jsonify(stats)
    
    @app.route('/api/models')
    def list_models():
        models = db.list_models()
        return jsonify([asdict(m) for m in models])
    
    @app.route('/api/models/<int:version>')
    def get_model(version):
        model = db.get_model_by_version(version)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        return jsonify(asdict(model))
    
    @app.route('/api/models/<int:version>/activate', methods=['POST'])
    def activate_model_api(version):
        success = registry.activate_model(version)
        if success:
            engine.load_active_model()
            return jsonify({'message': f'Model v{version} activated', 'version': version})
        else:
            return jsonify({'error': 'Failed to activate model'}), 500
    
    @app.route('/api/models/<int:version>', methods=['DELETE'])
    def delete_model(version):
        active = db.get_active_model()
        if active and active.version == version:
            return jsonify({'error': 'Cannot delete active model'}), 400
        
        model_path = registry.get_model_path(version)
        if model_path.exists():
            try:
                model_path.unlink()
            except Exception as e:
                logging.error(f"Failed to delete model file: {e}")
        
        with db.lock:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM models WHERE version = ?', (version,))
            conn.commit()
        
        return jsonify({'message': f'Model v{version} deleted'})
    
    @app.route('/api/models/upload', methods=['POST'])
    def upload_model():
        if 'model' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        file = request.files['model']
        notes = request.form.get('notes', '')
        
        if not file.filename.endswith('.eim'):
            return jsonify({'error': 'Only .eim files are supported'}), 400
        
        temp_path = Config.UPLOAD_DIR / secure_filename(file.filename)
        file.save(temp_path)
        
        try:
            version = registry.register_model(temp_path, notes)
            registry.activate_model(version)
            engine.load_active_model()
            temp_path.unlink()
            return jsonify({
                'message': f'Model v{version} deployed',
                'version': version
            })
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        use_async = request.form.get('async', 'false').lower() == 'true'
        
        filename = secure_filename(file.filename)
        timestamp = int(time.time() * 1000)
        image_path = Config.UPLOAD_DIR / f"{timestamp}_{filename}"
        file.save(image_path)
        
        try:
            if use_async and async_queue.enabled:
                output_path = Config.OUTPUT_DIR / f"result_{image_path.name}"
                success = async_queue.enqueue(image_path, output_path)
                
                if success:
                    return jsonify({
                        'message': 'Job queued for async processing',
                        'async': True,
                        'queue_size': async_queue.get_queue_size()
                    })
                else:
                    use_async = False
            
            if not use_async:
                result = engine.predict(image_path)
                output_path = Config.OUTPUT_DIR / f"result_{image_path.name}"
                engine.visualize_detections(image_path, result, output_path)
                
                return jsonify({
                    'model_version': result.model_version,
                    'image_name': result.image_name,
                    'timestamp': result.timestamp,
                    'detections': result.detections,
                    'inference_time': result.inference_time,
                    'image_url': f'/output/{output_path.name}',
                    'async': False
                })
                
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            if image_path.exists():
                try:
                    image_path.unlink()
                except:
                    pass
            return jsonify({'error': str(e)}), 500
    
    @app.route('/output/<filename>')
    def serve_output(filename):
        try:
            return send_file(Config.OUTPUT_DIR / filename)
        except Exception as e:
            return jsonify({'error': 'File not found'}), 404
    
    @app.route('/metrics')
    def metrics():
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    @app.route('/health')
    def health():
        active = db.get_active_model()
        return jsonify({
            'status': 'healthy' if active else 'no_model',
            'active_model': active.version if active else None,
            'redis_available': async_queue.enabled,
            'queue_size': async_queue.get_queue_size(),
            'timestamp': datetime.now().isoformat()
        })
    
    return app

def setup_logging():
    Config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Config.BASE_DIR / 'mlops.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    print("MLOps PCB Detection System - Starting...")
    
    setup_logging()
    Config.ensure_directories()
    
    logging.info("🚀 Starting MLOps PCB Detection System...")
    
    db = Database(Config.DB_PATH)
    model_registry = ModelRegistry(db, Config.MODEL_REGISTRY_DIR)
    inference_engine = InferenceEngine(db, model_registry)
    async_queue = AsyncQueue(inference_engine)
    performance_monitor = PerformanceMonitor(db, inference_engine, model_registry)
    
    if async_queue.enabled:
        async_queue.start_worker()
    else:
        logging.warning("⚠️ Async queue not started (Redis unavailable)")
    
    performance_monitor.start()
    
    observer = start_deployment_watcher(db, model_registry, inference_engine)
    
    app = create_app(db, model_registry, inference_engine, async_queue)
    
    active = db.get_active_model()
    if active:
        logging.info(f"🎯 Active model in database: v{active.version}")
        model_path = model_registry.get_model_path(active.version)
        if model_path.exists():
            logging.info(f"✓ Model file exists: {model_path}")
        else:
            logging.error(f"❌ Model file missing: {model_path}")
    else:
        logging.warning("⚠️ No active model in database. Deploy a model to start.")
    
    logging.info(f"🌐 Web UI: http://{Config.API_HOST}:{Config.API_PORT}")
    logging.info(f"📊 Metrics: http://{Config.API_HOST}:{Config.API_PORT}/metrics")
    logging.info(f"💚 Health: http://{Config.API_HOST}:{Config.API_PORT}/health")
    
    print(f"""
    Dashboard:  http://localhost:{Config.API_PORT}
    Metrics:    http://localhost:{Config.API_PORT}/metrics
    Deploy:     {Config.DEPLOYMENT_WATCH_DIR}
    
    Status:
    ✓ Database initialized
    ✓ Model registry ready
    ✓ Inference engine loaded
    {'✓ Redis queue active' if async_queue.enabled else '⚠ Redis unavailable'}
    ✓ Performance monitor running
    ✓ Auto-deployment watcher active
    
    Ready! 🚀
    """)
    
    try:
        app.run(
            host=Config.API_HOST,
            port=Config.API_PORT,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logging.info("\n🛑 Shutting down...")
        observer.stop()
        observer.join()
        async_queue.stop_worker()
        performance_monitor.stop()
        logging.info("✓ Shutdown complete")

if __name__ == '__main__':
    main()
