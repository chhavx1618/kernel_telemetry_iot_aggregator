# MLOps PCB Detection System - Complete Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [API Reference](#api-reference)
7. [Monitoring & Metrics](#monitoring--metrics)
8. [Deployment Workflows](#deployment-workflows)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 🎯 Overview

The MLOps PCB Detection System is a production-grade, end-to-end machine learning operations platform designed specifically for PCB (Printed Circuit Board) defect detection on edge devices like Raspberry Pi 4.

### Key Features

- 🤖 **Edge Impulse Integration**: Native support for .eim model files
- 📦 **Model Registry**: Version-controlled model storage and management
- 🔄 **Auto-Deployment**: Watch folder for automatic model deployment
- 📊 **Real-time Monitoring**: Prometheus metrics and performance tracking
- 🎯 **Inference API**: RESTful API with sync operations
- 🗄️ **SQLite Database**: Persistent storage for models, inferences, and metrics
- 🔙 **Automatic Rollback**: Performance-based model rollback system
- 🌐 **Web Dashboard**: Beautiful UI for model management and inference
- 📈 **Drift Detection**: Automatic data drift monitoring
- 🚨 **Health Checks**: Built-in system health monitoring

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Dashboard (Flask)                    │
│  - Model Management  - Inference UI  - Metrics Viewer       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                      REST API Layer                          │
│  /api/models  /api/predict  /metrics  /health               │
└────────┬─────────────┬──────────────┬────────────────┬──────┘
         │             │              │                │
    ┌────▼────┐  ┌────▼─────┐  ┌────▼────┐    ┌─────▼──────┐
    │ Model   │  │Inference │  │ Perf.   │    │ Deployment │
    │Registry │  │ Engine   │  │ Monitor │    │  Watcher   │
    └────┬────┘  └────┬─────┘  └────┬────┘    └─────┬──────┘
         │            │             │                │
         └────────────┴─────────────┴────────────────┘
                           │
                  ┌────────▼─────────┐
                  │  SQLite Database │
                  │  - Models        │
                  │  - Inferences    │
                  │  - Metrics       │
                  │  - Events        │
                  └──────────────────┘
```

### Data Flow

1. **Model Upload** → Model Registry → Database → Activation → Inference Engine
2. **Inference Request** → Inference Engine → Edge Impulse Runner → Results → Database
3. **Performance Monitoring** → Recent Inferences → Metrics Calculation → Rollback Decision
4. **Auto-Deployment** → File Watcher → Model Registration → Activation → Engine Reload

---

## 📥 Installation

### Prerequisites

```bash
# System Requirements
- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi OS (64-bit recommended)
- Python 3.8+
- 8GB+ storage space

# Required packages
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libatlas-base-dev \
    libportaudio2
```

### Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install flask
pip install opencv-python
pip install prometheus-client
pip install watchdog
pip install edge-impulse-linux
```

### Optional: Redis for Async Queue

```bash
# Install Redis (optional)
sudo apt install redis-server -y
pip install redis

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Quick Start

```bash
# Clone/download the script
wget https://your-url/mlops_pcb_system.py

# Make executable
chmod +x mlops_pcb_system.py

# Run
python3 mlops_pcb_system.py
```

---

## ⚙️ Configuration

### Config Class

All system configuration is centralized in the `Config` class:

```python
class Config:
    # Directory Structure
    BASE_DIR = Path("/opt/mlops-pcb")              # Root directory
    MODEL_REGISTRY_DIR = BASE_DIR / "models"       # Model storage
    DEPLOYMENT_WATCH_DIR = BASE_DIR / "deploy"     # Auto-deploy folder
    UPLOAD_DIR = BASE_DIR / "uploads"              # Image uploads
    OUTPUT_DIR = BASE_DIR / "output"               # Inference results
    DB_PATH = BASE_DIR / "mlops.db"                # SQLite database
    
    # Model Settings
    DEFAULT_MODEL = "model.eim"                    # Default model filename
    CONFIDENCE_THRESHOLD = 0.5                     # Min confidence for detections
    
    # Performance Thresholds
    DRIFT_DETECTION_WINDOW = 100                   # Number of inferences for drift
    ACCURACY_ROLLBACK_THRESHOLD = 0.7              # Min accuracy before rollback
    PERFORMANCE_DEGRADATION_THRESHOLD = 2.0        # Max inference time multiplier
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # API Server
    API_HOST = "0.0.0.0"                          # Listen on all interfaces
    API_PORT = 8080                                # API server port
    METRICS_PORT = 9090                            # Prometheus metrics port
```

### Environment Variables

```bash
# Override Redis settings
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

# Run the system
python3 mlops_pcb_system.py
```

### Directory Structure

After initialization, the system creates:

```
/opt/mlops-pcb/
├── models/                    # Model registry
│   ├── model_v1.eim
│   ├── model_v2.eim
│   └── model_v3.eim
├── deploy/                    # Watch folder for auto-deployment
│   └── processed/             # Processed models moved here
├── uploads/                   # Uploaded images (temp storage)
├── output/                    # Inference results with visualizations
├── mlops.db                   # SQLite database
└── mlops.log                  # Application logs
```

---

## 🔧 Core Components

### 1. Database (`Database` class)

**Purpose**: Manages all persistent storage using SQLite

**Tables**:

```sql
-- Models table: Stores model metadata
CREATE TABLE models (
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
);

-- Inferences table: Stores every inference result
CREATE TABLE inferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version INTEGER NOT NULL,
    image_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    detections TEXT NOT NULL,           -- JSON array
    inference_time REAL NOT NULL,
    confidence_scores TEXT NOT NULL,    -- JSON array
    FOREIGN KEY (model_version) REFERENCES models (version)
);

-- Metrics table: Aggregated performance metrics
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    accuracy REAL,
    avg_confidence REAL,
    avg_inference_time REAL,
    detection_count INTEGER,
    FOREIGN KEY (model_version) REFERENCES models (version)
);

-- Deployment events: Audit trail
CREATE TABLE deployment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version INTEGER NOT NULL,
    event_type TEXT NOT NULL,           -- ACTIVATE, ROLLBACK, etc.
    timestamp TEXT NOT NULL,
    reason TEXT,
    previous_version INTEGER,
    FOREIGN KEY (model_version) REFERENCES models (version)
);
```

**Key Methods**:

```python
# Model management
db.register_model(filename, file_hash, notes)  # Register new model
db.activate_model(version)                      # Set active model
db.get_active_model()                           # Get currently active model
db.list_models()                                # List all models

# Inference tracking
db.log_inference(result)                        # Log inference result
db.get_recent_inferences(version, limit)        # Get recent inferences

# Performance monitoring
db.log_metrics(metrics)                         # Log performance metrics
db.log_deployment_event(version, type, reason)  # Log deployment events
```

**Thread Safety**: Uses `threading.Lock()` for safe concurrent access

---

### 2. Model Registry (`ModelRegistry` class)

**Purpose**: Manages model files and version control

**Features**:
- SHA256 hash-based deduplication
- Safe atomic file operations
- Version numbering
- File permission management

**Key Methods**:

```python
# Register a new model
version = registry.register_model(source_path, notes="Production model")
# Returns: version number (int)
# - Computes SHA256 hash
# - Checks for duplicates
# - Copies file to registry
# - Makes file executable
# - Registers in database

# Activate a model version
success = registry.activate_model(version)
# - Deactivates current model
# - Sets new model as active
# - Logs deployment event
# - Updates Prometheus metrics

# Get model file path
path = registry.get_model_path(version)
# Returns: Path to model_vN.eim file
```

**File Operations**:

```python
def safe_copy_file(source, dest):
    """
    Atomically copy file with proper permissions:
    1. Read source data
    2. Write to temporary file
    3. Set executable permissions (755)
    4. Atomic rename to destination
    """
```

---

### 3. Inference Engine (`InferenceEngine` class)

**Purpose**: Executes ML inference using Edge Impulse models

**Architecture**:
```
┌─────────────────────────────────────────┐
│         Inference Engine                │
├─────────────────────────────────────────┤
│ - runner: ImageImpulseRunner            │
│ - model_info: Dict (model metadata)     │
│ - current_version: int                  │
│ - lock: threading.Lock()                │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Edge Impulse Runner                │
│  - Load .eim model                      │
│  - Extract features from image          │
│  - Run inference                        │
│  - Return detections                    │
└─────────────────────────────────────────┘
```

**Key Methods**:

```python
# Load a model
engine.load_model(version, model_path)
# - Stops existing runner
# - Initializes ImageImpulseRunner
# - Validates model info
# - Sets permissions if needed

# Run inference
result = engine.predict(image_path, confidence_threshold=0.5)
# Returns: InferenceResult with:
#   - model_version
#   - image_name
#   - timestamp
#   - detections: [{'label', 'confidence', 'bbox'}]
#   - inference_time
#   - confidence_scores

# Visualize detections
engine.visualize_detections(image_path, result, output_path)
# - Draws bounding boxes
# - Adds labels with confidence
# - Saves annotated image
```

**Thread Safety**: Uses lock for concurrent inference requests

---

### 4. Performance Monitor (`PerformanceMonitor` class)

**Purpose**: Continuous monitoring and automatic rollback

**Monitoring Loop** (runs every 60 seconds):

```python
def _monitor_loop():
    while running:
        _check_performance()    # Accuracy monitoring
        _detect_drift()         # Data drift detection
        sleep(60)
```

**Performance Checks**:

```python
def _check_performance():
    """
    1. Get last N inferences (default: 100)
    2. Calculate metrics:
       - Average confidence
       - Average inference time
       - Detection count
    3. Update Prometheus gauges
    4. Check accuracy threshold
    5. Trigger rollback if needed
    """
```

**Drift Detection**:

```python
def _detect_drift():
    """
    1. Get recent confidence scores
    2. Calculate variance
    3. Compare to threshold
    4. Alert if drift detected
    
    Formula: variance = Σ(x - mean)² / n
    Threshold: variance > 0.15
    """
```

**Automatic Rollback**:

```python
def _trigger_rollback(reason):
    """
    1. Find previous deployed model
    2. Activate previous model
    3. Log rollback event
    4. Reload inference engine
    5. Update metrics
    """
```

**Rollback Conditions**:
- Accuracy < 70% (configurable)
- Inference time > 2x baseline
- Manual trigger via API

---

### 5. Deployment Watcher (`DeploymentWatcher` class)

**Purpose**: Automatic model deployment via filesystem monitoring

**How It Works**:

```
1. Watch /opt/mlops-pcb/deploy/ directory
2. On new .eim file detected:
   ├─ Wait for file to be fully written
   ├─ Compute SHA256 hash
   ├─ Check for duplicates
   ├─ Register in model registry
   ├─ Activate new model
   ├─ Reload inference engine
   └─ Move to processed/ folder
```

**Implementation**:

```python
class DeploymentWatcher(FileSystemEventHandler):
    def on_created(self, event):
        """
        Triggered when new file is created
        - Filters for .eim files only
        - Prevents duplicate processing
        - Handles deployment workflow
        """
```

**File Ready Check**:

```python
def _wait_for_file_ready(filepath, timeout=10):
    """
    Wait for file to finish writing:
    1. Check file size every 0.5s
    2. If size stable for 1s, file is ready
    3. Timeout after 10s
    """
```

**Usage**:

```bash
# Deploy a new model
cp new_model.eim /opt/mlops-pcb/deploy/

# System automatically:
# - Detects the file
# - Registers as new version
# - Activates it
# - Starts using it
# - Moves to deploy/processed/
```

---

### 6. Async Queue (`AsyncQueue` class)

**Purpose**: Redis-based asynchronous inference processing (optional)

**Note**: Currently disabled in UI but functional in backend

**Architecture**:

```
┌──────────────┐      ┌─────────────┐      ┌────────────────┐
│ API Request  │─────▶│ Redis Queue │─────▶│ Worker Thread  │
└──────────────┘      └─────────────┘      └────────────────┘
                            │                       │
                            │                       ▼
                            │              ┌─────────────────┐
                            │              │ Inference Engine│
                            │              └─────────────────┘
                            │                       │
                            │                       ▼
                            │              ┌─────────────────┐
                            └─────────────▶│ Save Results    │
                                           └─────────────────┘
```

**Key Methods**:

```python
# Enqueue a job
success = async_queue.enqueue(image_path, output_path)
# - Serializes job to JSON
# - Pushes to Redis list
# - Returns success status

# Worker processes jobs
def _worker_loop():
    while running:
        job = redis.blpop('inference_queue', timeout=1)
        if job:
            _process_job(job)
```

**Job Format**:

```json
{
    "image_path": "/opt/mlops-pcb/uploads/image.jpg",
    "output_path": "/opt/mlops-pcb/output/result_image.jpg",
    "timestamp": "2025-11-06T01:00:00.000000"
}
```

---

## 🌐 API Reference

### Base URL
```
http://localhost:8080
```

### Endpoints

#### 1. Dashboard

```http
GET /
```

Returns the web dashboard HTML interface.

---

#### 2. System Statistics

```http
GET /api/stats
```

**Response**:
```json
{
    "active_version": 1,
    "total_inferences": 150,
    "total_detections": 1250,
    "avg_confidence": 0.85,
    "queue_size": 0
}
```

---

#### 3. Health Check

```http
GET /health
```

**Response**:
```json
{
    "status": "healthy",
    "active_model": 1,
    "redis_available": true,
    "queue_size": 0,
    "timestamp": "2025-11-06T01:00:00.000000"
}
```

**Status Values**:
- `healthy`: System operational with active model
- `no_model`: No model loaded
- `degraded`: Model loaded but issues detected

---

#### 4. List Models

```http
GET /api/models
```

**Response**:
```json
[
    {
        "version": 2,
        "filename": "model.eim",
        "file_hash": "abc123...",
        "upload_date": "2025-11-06T00:00:00.000000",
        "deployed_date": "2025-11-06T00:05:00.000000",
        "is_active": true,
        "accuracy": 0.85,
        "avg_inference_time": 0.234,
        "total_inferences": 150,
        "total_detections": 1250,
        "notes": "Production model"
    },
    {
        "version": 1,
        "filename": "old_model.eim",
        "is_active": false,
        ...
    }
]
```

---

#### 5. Get Model Details

```http
GET /api/models/{version}
```

**Parameters**:
- `version` (path): Model version number

**Response**: Same as single model in list response

**Errors**:
- `404`: Model not found

---

#### 6. Upload Model

```http
POST /api/models/upload
```

**Content-Type**: `multipart/form-data`

**Parameters**:
- `model` (file): .eim model file (required)
- `notes` (string): Optional notes

**Response**:
```json
{
    "message": "Model v3 deployed",
    "version": 3
}
```

**Errors**:
- `400`: No file provided or invalid file type
- `500`: Deployment failed

---

#### 7. Activate Model

```http
POST /api/models/{version}/activate
```

**Parameters**:
- `version` (path): Model version to activate

**Response**:
```json
{
    "message": "Model v2 activated",
    "version": 2
}
```

**Errors**:
- `500`: Activation failed

---

#### 8. Delete Model

```http
DELETE /api/models/{version}
```

**Parameters**:
- `version` (path): Model version to delete

**Response**:
```json
{
    "message": "Model v1 deleted"
}
```

**Errors**:
- `400`: Cannot delete active model

---

#### 9. Run Inference

```http
POST /api/predict
```

**Content-Type**: `multipart/form-data`

**Parameters**:
- `image` (file): Image file (JPG, PNG, BMP)
- `async` (string): "true" for async processing (optional, requires Redis)

**Sync Response**:
```json
{
    "model_version": 2,
    "image_name": "pcb_board.jpg",
    "timestamp": "2025-11-06T01:00:00.000000",
    "detections": [
        {
            "label": "scratch",
            "confidence": 0.92,
            "bbox": [150, 200, 50, 30]
        },
        {
            "label": "missing_component",
            "confidence": 0.88,
            "bbox": [300, 150, 40, 40]
        }
    ],
    "inference_time": 0.234,
    "image_url": "/output/result_pcb_board.jpg",
    "async": false
}
```

**Async Response**:
```json
{
    "message": "Job queued for async processing",
    "async": true,
    "queue_size": 3
}
```

**Errors**:
- `400`: No image provided
- `500`: Inference failed

---

#### 10. Get Result Image

```http
GET /output/{filename}
```

**Parameters**:
- `filename` (path): Result image filename from inference response

Returns the image file with bounding boxes drawn.

---

#### 11. Prometheus Metrics

```http
GET /metrics
```

Returns Prometheus-formatted metrics (see Monitoring section).

---

## 📊 Monitoring & Metrics

### Prometheus Metrics

The system exports metrics in Prometheus format at `/metrics`.

#### Counters

```python
# Total inference requests
pcb_inferences_total{model_version="v2", status="success"} 150

# Total detections by type
pcb_detections_total{model_version="v2", label="scratch"} 45
pcb_detections_total{model_version="v2", label="missing_component"} 30

# Model deployments
pcb_model_deployments_total{status="success"} 5
pcb_model_deployments_total{status="failed"} 1

# Rollback events
pcb_model_rollbacks_total 2
```

#### Histograms

```python
# Inference latency distribution
pcb_inference_latency_seconds_bucket{model_version="v2", le="0.1"} 50
pcb_inference_latency_seconds_bucket{model_version="v2", le="0.5"} 140
pcb_inference_latency_seconds_bucket{model_version="v2", le="1.0"} 150
pcb_inference_latency_seconds_sum{model_version="v2"} 35.1
pcb_inference_latency_seconds_count{model_version="v2"} 150

# Confidence score distribution
pcb_confidence_score_bucket{model_version="v2", le="0.7"} 10
pcb_confidence_score_bucket{model_version="v2", le="0.9"} 120
pcb_confidence_score_bucket{model_version="v2", le="1.0"} 150
```

#### Gauges

```python
# Currently active model version
pcb_active_model_version 2

# Model accuracy (last N predictions)
pcb_model_accuracy{model_version="v2"} 0.85

# Data drift score
pcb_drift_score{model_version="v2"} 0.08

# Redis queue size
pcb_inference_queue_size 3
```

### Grafana Integration

Example Grafana dashboard queries:

```promql
# Inference rate (per second)
rate(pcb_inferences_total[5m])

# Average inference latency
rate(pcb_inference_latency_seconds_sum[5m]) / rate(pcb_inference_latency_seconds_count[5m])

# Model accuracy over time
pcb_model_accuracy

# Detection distribution
sum by (label) (pcb_detections_total)

# P95 latency
histogram_quantile(0.95, rate(pcb_inference_latency_seconds_bucket[5m]))
```

---

## 🚀 Deployment Workflows

### Method 1: Auto-Deployment (Recommended)

```bash
# Copy model to watch folder
cp new_model.eim /opt/mlops-pcb/deploy/

# System automatically deploys and activates
# Check logs for confirmation
tail -f /opt/mlops-pcb/mlops.log
```

**Output**:
```
📦 New model detected: new_model.eim
✓ Model registered: v3 (new_model.eim)
✓ Model v3 activated
✅ Model v3 loaded successfully: PCB Detection
✓ Model v3 deployment complete
```

---

### Method 2: Web UI Upload

1. Open dashboard: `http://localhost:8080`
2. Navigate to "🚀 Deploy" tab
3. Click upload area or drag & drop
4. Add notes (optional)
5. Click "🚀 Deploy"

---

### Method 3: API Upload

```bash
curl -X POST http://localhost:8080/api/models/upload \
  -F "model=@new_model.eim" \
  -F "notes=Production model v3"
```

---

### Method 4: Manual Registration

```python
from pathlib import Path
from mlops_system import Database, ModelRegistry

db = Database(Path("/opt/mlops-pcb/mlops.db"))
registry = ModelRegistry(db, Path("/opt/mlops-pcb/models"))

# Register model
version = registry.register_model(
    Path("path/to/model.eim"),
    notes="Manual deployment"
)

# Activate it
registry.activate_model(version)
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Symptom**:
```
❌ Failed to load model v1: Model file "..." is not executable
```

**Solution**:
```bash
# Fix permissions
chmod +x /opt/mlops-pcb/models/model_v1.eim

# Or let the system fix it automatically (restart)
```

---

#### 2. No Active Model

**Symptom**:
```
⚠️ No active model found in database
```

**Solution**:
```bash
# Deploy a model using any method above
# Or activate existing model via UI/API
```

---

#### 3. Redis Connection Failed

**Symptom**:
```
⚠️ Redis connection failed: Connection refused
Make sure Redis is running: sudo systemctl status redis-server
```

**Solution**:
```bash
# Install Redis
sudo apt install redis-server -y

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Restart the system
```

Note: System works without Redis, just without async queue.

---

#### 4. Inference Failed

**Symptom**:
```
❌ Model info not available
❌ Prediction error: Model not properly initialized
```

**Solution**:
```bash
# Check if model file exists
ls -la /opt/mlops-pcb/models/

# Check database for active model
sqlite3 /opt/mlops-pcb/mlops.db "SELECT * FROM models WHERE is_active=1;"

# Restart the system to reload
```

---

#### 5. Permission Denied

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: '/opt/mlops-pcb/...'
```

**Solution**:
```bash
# Fix ownership
sudo chown -R $USER:$USER /opt/mlops-pcb/

# Or run with sudo
sudo -E venv/bin/python3 mlops_system.py
```

---

#### 6. Out of Memory

**Symptom**:
System crashes or becomes unresponsive during inference

**Solution**:
```bash
# Check memory usage
free -h

# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Restart system
sudo reboot
```

---

### Diagnostic Commands

```bash
# Check system status
curl http://localhost:8080/health

# View recent logs
tail -100 /opt/mlops-pcb/mlops.log

# Check database
sqlite3 /opt/mlops-pcb/mlops.db
> SELECT * FROM models;
> SELECT * FROM inferences ORDER BY timestamp DESC LIMIT 10;
> SELECT * FROM deployment_events;
> .exit

# Check file permissions
ls -la /opt/mlops-pcb/models/

# Monitor system resources
htop

# Check port usage
sudo netstat -tulpn | grep 8080

# Test Edge Impulse
python3 -c "from edge_impulse_linux.image import ImageImpulseRunner; print('OK')"
```

---

## 🎯 Best Practices

### 1. Model Management

```bash
# Always test models before deployment
# Use descriptive version notes
# Keep at least 2 previous versions
# Monitor accuracy after deployment
# Set up rollback thresholds appropriately
```

### 2. Performance Optimization

```python
# Adjust confidence threshold based on use case
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections, more false positives

# Tune monitoring windows
DRIFT_DETECTION_WINDOW = 100  # Larger = more stable, slower detection

# Set realistic rollback thresholds
ACCURACY_ROLLBACK_THRESHOLD = 0.7  # Based on your model's baseline
```

### 3. Production Deployment

```bash
# Run as system service
sudo nano /etc/systemd/system/mlops-pcb.service

[Unit]
Description=MLOps PCB Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pcb-detector
Environment="PATH=/home/pi/pcb-detector/venv/bin"
ExecStart=/home/pi/pcb-detector/venv/bin/python3 mlops_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mlops-pcb
sudo systemctl start mlops-pcb

# Check status
sudo systemctl status mlops-pcb
```

### 4. Backup Strategy

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backup/mlops-pcb-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup database
cp /opt/mlops-pcb/mlops.db $BACKUP_DIR/

# Backup active model
cp /opt/mlops-pcb/models/model_v*.eim $BACKUP_DIR/

# Backup logs
cp /opt/mlops-pcb/mlops.log $BACKUP_DIR/

echo "Backup complete: $BACKUP_DIR"

# Schedule with cron
# crontab -e
# 0 2 * * * /home/pi/backup-mlops.sh
```

### 5. Monitoring Setup

```bash
# Install Prometheus (optional)
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-arm64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Configure prometheus.yml
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlops-pcb'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
EOF

# Start Prometheus
./prometheus --config.file=prometheus.yml

# Access at http://localhost:9090
```

### 6. Security Considerations

```bash
# Restrict API access
# Use nginx as reverse proxy with authentication

# Example nginx config
server {
    listen 80;
    server_name pcb-detector.local;
    
    location / {
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:8080;
    }
}

# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Firewall rules
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw deny 8080/tcp  # Block direct access
sudo ufw enable
```

### 7. Data Management

```bash
# Clean up old uploads (run daily)
find /opt/mlops-pcb/uploads -type f -mtime +7 -delete

# Archive old inference results
find /opt/mlops-pcb/output -type f -mtime +30 -exec gzip {} \;
find /opt/mlops-pcb/output -name "*.gz" -mtime +90 -delete

# Vacuum database periodically
sqlite3 /opt/mlops-pcb/mlops.db "VACUUM;"

# Prune old inferences from database
sqlite3 /opt/mlops-pcb/mlops.db \
  "DELETE FROM inferences WHERE timestamp < datetime('now', '-90 days');"
```

---

## 📈 Advanced Features

### Custom Metrics Integration

```python
# Add custom Prometheus metrics
from prometheus_client import Counter, Histogram

# Define custom metrics
custom_defect_counter = Counter(
    'pcb_custom_defects_total',
    'Custom defect tracking',
    ['severity', 'location']
)

# Use in your code
def classify_defect_severity(detection):
    if detection['confidence'] > 0.9:
        severity = 'critical'
    elif detection['confidence'] > 0.7:
        severity = 'high'
    else:
        severity = 'medium'
    
    custom_defect_counter.labels(
        severity=severity,
        location=detection['label']
    ).inc()
```

### Webhook Notifications

```python
import requests

def send_webhook_notification(event_type, data):
    """Send notifications on important events"""
    webhook_url = "https://your-webhook-url.com/notify"
    
    payload = {
        'event': event_type,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Webhook failed: {e}")

# Use for rollbacks
def _trigger_rollback(self, reason):
    # ... existing rollback code ...
    
    send_webhook_notification('rollback', {
        'from_version': self.engine.current_version,
        'to_version': previous.version,
        'reason': reason
    })
```

### Batch Inference

```python
def batch_inference(image_dir: Path, output_dir: Path):
    """
    Run inference on multiple images
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory for results
    """
    results = []
    
    for image_path in image_dir.glob("*.jpg"):
        try:
            result = engine.predict(image_path)
            output_path = output_dir / f"result_{image_path.name}"
            engine.visualize_detections(image_path, result, output_path)
            results.append(result)
            
        except Exception as e:
            logging.error(f"Failed on {image_path.name}: {e}")
            continue
    
    # Generate report
    report = {
        'total_images': len(results),
        'total_detections': sum(len(r.detections) for r in results),
        'avg_confidence': sum(
            sum(r.confidence_scores) / len(r.confidence_scores) 
            for r in results if r.confidence_scores
        ) / len(results),
        'avg_inference_time': sum(r.inference_time for r in results) / len(results)
    }
    
    return results, report

# Usage
results, report = batch_inference(
    Path("/data/pcb-images"),
    Path("/data/results")
)
```

### Model A/B Testing

```python
class ABTestingEngine:
    """Compare two models side-by-side"""
    
    def __init__(self, model_a_version: int, model_b_version: int):
        self.engine_a = InferenceEngine(db, registry)
        self.engine_b = InferenceEngine(db, registry)
        
        self.engine_a.load_model(model_a_version, 
                                 registry.get_model_path(model_a_version))
        self.engine_b.load_model(model_b_version,
                                 registry.get_model_path(model_b_version))
    
    def compare(self, image_path: Path):
        """Run both models and compare results"""
        result_a = self.engine_a.predict(image_path)
        result_b = self.engine_b.predict(image_path)
        
        comparison = {
            'model_a': {
                'version': result_a.model_version,
                'detections': len(result_a.detections),
                'avg_confidence': sum(result_a.confidence_scores) / len(result_a.confidence_scores) if result_a.confidence_scores else 0,
                'inference_time': result_a.inference_time
            },
            'model_b': {
                'version': result_b.model_version,
                'detections': len(result_b.detections),
                'avg_confidence': sum(result_b.confidence_scores) / len(result_b.confidence_scores) if result_b.confidence_scores else 0,
                'inference_time': result_b.inference_time
            }
        }
        
        return result_a, result_b, comparison
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Model to Production

on:
  push:
    paths:
      - 'models/*.eim'
  workflow_dispatch:

jobs:
  deploy:
    runs-on: self-hosted
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Validate model file
        run: |
          if [ ! -f models/*.eim ]; then
            echo "No model file found"
            exit 1
          fi
          
      - name: Deploy to production
        run: |
          MODEL_FILE=$(ls models/*.eim | head -1)
          curl -X POST http://localhost:8080/api/models/upload \
            -F "model=@$MODEL_FILE" \
            -F "notes=Auto-deployed from CI/CD"
      
      - name: Wait for deployment
        run: sleep 10
      
      - name: Health check
        run: |
          HEALTH=$(curl -s http://localhost:8080/health)
          STATUS=$(echo $HEALTH | jq -r '.status')
          
          if [ "$STATUS" != "healthy" ]; then
            echo "Deployment failed - system not healthy"
            exit 1
          fi
      
      - name: Run smoke tests
        run: |
          # Upload test image
          RESULT=$(curl -X POST http://localhost:8080/api/predict \
            -F "image=@test_images/test_pcb.jpg")
          
          # Check for detections
          DETECTIONS=$(echo $RESULT | jq '.detections | length')
          
          if [ $DETECTIONS -lt 1 ]; then
            echo "Warning: No detections on test image"
          fi
      
      - name: Notify on failure
        if: failure()
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -H 'Content-Type: application/json' \
            -d '{"text":"Model deployment failed!"}'
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Validate Model') {
            steps {
                script {
                    sh '''
                        if [ ! -f models/*.eim ]; then
                            echo "No model file found"
                            exit 1
                        fi
                    '''
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                script {
                    sh '''
                        MODEL_FILE=$(ls models/*.eim | head -1)
                        scp $MODEL_FILE pi@staging:/opt/mlops-pcb/deploy/
                    '''
                }
            }
        }
        
        stage('Staging Tests') {
            steps {
                script {
                    sh '''
                        # Wait for deployment
                        sleep 15
                        
                        # Run tests
                        ssh pi@staging "cd /home/pi/tests && python3 test_inference.py"
                    '''
                }
            }
        }
        
        stage('Deploy to Production') {
            input {
                message "Deploy to production?"
                ok "Deploy"
            }
            steps {
                script {
                    sh '''
                        MODEL_FILE=$(ls models/*.eim | head -1)
                        scp $MODEL_FILE pi@production:/opt/mlops-pcb/deploy/
                    '''
                }
            }
        }
        
        stage('Production Validation') {
            steps {
                script {
                    sh '''
                        sleep 15
                        
                        # Check health
                        HEALTH=$(curl -s http://production:8080/health)
                        echo $HEALTH | jq .
                        
                        STATUS=$(echo $HEALTH | jq -r '.status')
                        if [ "$STATUS" != "healthy" ]; then
                            exit 1
                        fi
                    '''
                }
            }
        }
    }
    
    post {
        failure {
            emailext(
                subject: "Model Deployment Failed",
                body: "Check Jenkins for details",
                to: "team@company.com"
            )
        }
        success {
            emailext(
                subject: "Model Deployed Successfully",
                body: "New model is live in production",
                to: "team@company.com"
            )
        }
    }
}
```

---

## 🧪 Testing

### Unit Tests

```python
import unittest
from pathlib import Path
from mlops_system import Database, ModelRegistry, InferenceEngine

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db_path = Path("/tmp/test_mlops.db")
        self.db = Database(self.db_path)
    
    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_register_model(self):
        version = self.db.register_model(
            "test_model.eim",
            "abc123hash",
            "Test model"
        )
        self.assertEqual(version, 1)
    
    def test_activate_model(self):
        version = self.db.register_model("test.eim", "hash1", "")
        self.db.activate_model(version)
        
        active = self.db.get_active_model()
        self.assertEqual(active.version, version)
        self.assertTrue(active.is_active)

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.db_path = Path("/tmp/test_mlops.db")
        self.registry_dir = Path("/tmp/test_registry")
        self.registry_dir.mkdir(exist_ok=True)
        
        self.db = Database(self.db_path)
        self.registry = ModelRegistry(self.db, self.registry_dir)
    
    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()
        if self.registry_dir.exists():
            import shutil
            shutil.rmtree(self.registry_dir)
    
    def test_compute_hash(self):
        # Create test file
        test_file = Path("/tmp/test_model.eim")
        test_file.write_text("test content")
        
        hash1 = self.registry.compute_hash(test_file)
        hash2 = self.registry.compute_hash(test_file)
        
        self.assertEqual(hash1, hash2)
        test_file.unlink()

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
import requests
import time

class TestAPIEndpoints:
    BASE_URL = "http://localhost:8080"
    
    def test_health_check(self):
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['healthy', 'no_model', 'degraded']
    
    def test_model_upload_and_inference(self):
        # Upload model
        with open('test_model.eim', 'rb') as f:
            files = {'model': f}
            data = {'notes': 'Test model'}
            response = requests.post(
                f"{self.BASE_URL}/api/models/upload",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        version = response.json()['version']
        
        # Wait for activation
        time.sleep(2)
        
        # Run inference
        with open('test_image.jpg', 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{self.BASE_URL}/api/predict",
                files=files
            )
        
        assert response.status_code == 200
        result = response.json()
        assert result['model_version'] == version
        assert 'detections' in result
    
    def test_model_list(self):
        response = requests.get(f"{self.BASE_URL}/api/models")
        assert response.status_code == 200
        
        models = response.json()
        assert isinstance(models, list)

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
```

### Load Testing

```python
import concurrent.futures
import requests
import time

def run_inference(image_path, iteration):
    """Single inference request"""
    try:
        start = time.time()
        
        with open(image_path, 'rb') as f:
            response = requests.post(
                'http://localhost:8080/api/predict',
                files={'image': f},
                timeout=30
            )
        
        elapsed = time.time() - start
        
        return {
            'iteration': iteration,
            'status_code': response.status_code,
            'elapsed': elapsed,
            'success': response.status_code == 200
        }
    except Exception as e:
        return {
            'iteration': iteration,
            'status_code': 0,
            'elapsed': 0,
            'success': False,
            'error': str(e)
        }

def load_test(image_path, num_requests=100, concurrency=10):
    """
    Run load test
    
    Args:
        image_path: Path to test image
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
    """
    print(f"Starting load test: {num_requests} requests, {concurrency} concurrent")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_inference, image_path, i)
            for i in range(num_requests)
        ]
        
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    # Calculate stats
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    avg_latency = sum(r['elapsed'] for r in successful) / len(successful) if successful else 0
    min_latency = min(r['elapsed'] for r in successful) if successful else 0
    max_latency = max(r['elapsed'] for r in successful) if successful else 0
    
    throughput = len(successful) / total_time
    
    print(f"\nLoad Test Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Successful: {len(successful)}/{num_requests}")
    print(f"  Failed: {len(failed)}/{num_requests}")
    print(f"  Throughput: {throughput:.2f} req/s")
    print(f"  Latency:")
    print(f"    Average: {avg_latency:.3f}s")
    print(f"    Min: {min_latency:.3f}s")
    print(f"    Max: {max_latency:.3f}s")
    
    if failed:
        print(f"\nFailures:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"  Iteration {r['iteration']}: {r.get('error', 'Unknown error')}")

if __name__ == '__main__':
    load_test('test_image.jpg', num_requests=100, concurrency=10)
```

---

## 📊 Data Analysis

### Extract Inference History

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def analyze_model_performance(db_path: str, model_version: int):
    """Analyze model performance from database"""
    
    conn = sqlite3.connect(db_path)
    
    # Get inference history
    query = """
    SELECT 
        timestamp,
        inference_time,
        detections,
        confidence_scores
    FROM inferences
    WHERE model_version = ?
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, params=(model_version,))
    
    # Parse JSON fields
    import json
    df['detections'] = df['detections'].apply(json.loads)
    df['confidence_scores'] = df['confidence_scores'].apply(json.loads)
    df['num_detections'] = df['detections'].apply(len)
    df['avg_confidence'] = df['confidence_scores'].apply(
        lambda x: sum(x)/len(x) if x else 0
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Statistics
    print(f"Model v{model_version} Statistics:")
    print(f"  Total inferences: {len(df)}")
    print(f"  Avg inference time: {df['inference_time'].mean():.3f}s")
    print(f"  Avg detections per image: {df['num_detections'].mean():.1f}")
    print(f"  Avg confidence: {df['avg_confidence'].mean():.2%}")
    
    # Plot inference time over time
    plt.figure(figsize=(12, 4))
    plt.plot(df['timestamp'], df['inference_time'])
    plt.xlabel('Time')
    plt.ylabel('Inference Time (s)')
    plt.title(f'Model v{model_version} - Inference Time Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'model_v{model_version}_inference_time.png')
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    all_confidences = [c for scores in df['confidence_scores'] for c in scores]
    plt.hist(all_confidences, bins=50, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'Model v{model_version} - Confidence Score Distribution')
    plt.tight_layout()
    plt.savefig(f'model_v{model_version}_confidence_dist.png')
    
    conn.close()
    
    return df

# Usage
df = analyze_model_performance('/opt/mlops-pcb/mlops.db', model_version=2)
```

### Compare Model Versions

```python
def compare_models(db_path: str, version_a: int, version_b: int):
    """Compare two model versions"""
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        model_version,
        AVG(inference_time) as avg_time,
        COUNT(*) as total_inferences,
        SUM(json_array_length(detections)) as total_detections
    FROM inferences
    WHERE model_version IN (?, ?)
    GROUP BY model_version
    """
    
    df = pd.read_sql_query(query, conn, params=(version_a, version_b))
    
    print("\nModel Comparison:")
    print(df.to_string(index=False))
    
    # Statistical significance test
    # Get all inference times for each model
    times_a = pd.read_sql_query(
        "SELECT inference_time FROM inferences WHERE model_version = ?",
        conn, params=(version_a,)
    )['inference_time']
    
    times_b = pd.read_sql_query(
        "SELECT inference_time FROM inferences WHERE model_version = ?",
        conn, params=(version_b,)
    )['inference_time']
    
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(times_a, times_b)
    
    print(f"\nInference Time Comparison:")
    print(f"  Model v{version_a}: {times_a.mean():.3f}s ± {times_a.std():.3f}s")
    print(f"  Model v{version_b}: {times_b.mean():.3f}s ± {times_b.std():.3f}s")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  Result: Statistically significant difference")
    else:
        print("  Result: No significant difference")
    
    conn.close()

# Usage
compare_models('/opt/mlops-pcb/mlops.db', version_a=1, version_b=2)
```

---

## 🔐 Security Best Practices

### 1. Authentication

```python
# Add basic authentication to Flask app
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("your_secure_password_here")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# Protect routes
@app.route('/api/models/upload', methods=['POST'])
@auth.login_required
def upload_model():
    # ... existing code ...
```

### 2. Input Validation

```python
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate uploaded image"""
    # Check file extension
    if not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset
    
    if size > MAX_IMAGE_SIZE:
        raise ValueError("File too large")
    
    # Validate image format
    try:
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        file.seek(0)
        
        if img is None:
            raise ValueError("Invalid image")
    except:
        raise ValueError("Image validation failed")
```

### 3. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... existing code ...
```

### 4. Secure Configuration

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """Encrypted configuration management"""
    
    def __init__(self, key_file: Path):
        if key_file.exists():
            self.key = key_file.read_bytes()
        else:
            self.key = Fernet.generate_key()
            key_file.write_bytes(self.key)
            os.chmod(key_file, 0o600)
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, value: str) -> bytes:
        return self.cipher.encrypt(value.encode())
    
    def decrypt(self, value: bytes) -> str:
        return self.cipher.decrypt(value).decode()

# Usage
config = SecureConfig(Path("/opt/mlops-pcb/.key"))
encrypted_api_key = config.encrypt("your-api-key")
```

---

## 📚 Additional Resources

### Documentation Links

- **Edge Impulse**: https://docs.edgeimpulse.com/
- **Flask**: https://flask.palletsprojects.com/
- **Prometheus**: https://prometheus.io/docs/
- **OpenCV**: https://docs.opencv.org/
- **SQLite**: https://www.sqlite.org/docs.html

### Community & Support

- GitHub Issues: [Your repo URL]
- Email: support@yourcompany.com
- Documentation: [Your docs URL]

### License

This project is licensed under the MIT License.

### Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Changelog

#### v1.0.0 (2025-11-06)
- Initial release
- Model registry and version control
- Auto-deployment system
- Performance monitoring
- Prometheus metrics
- Web dashboard
- REST API
- SQLite database
- Automatic rollback

---

## 🎓 Glossary

**MLOps**: Machine Learning Operations - practices for deploying and maintaining ML systems in production

**Inference**: The process of using a trained model to make predictions on new data

**.eim file**: Edge Impulse Model file format

**Model Registry**: Centralized store for ML models with version control

**Drift Detection**: Monitoring for changes in data distribution over time

**Rollback**: Reverting to a previous model version

**Prometheus**: Open-source monitoring and alerting toolkit

**SQLite**: Lightweight embedded relational database

**Edge Device**: Computing device at the edge of a network (e.g., Raspberry Pi)

**Bounding Box**: Rectangle drawn around detected objects

**Confidence Score**: Model's certainty in its prediction (0-1)

**Throughput**: Number of inferences per second

**Latency**: Time taken to complete a single inference

---

## 📞 Support

For issues, questions, or contributions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs: `/opt/mlops-pcb/mlops.log`
3. Check system health: `curl http://localhost:8080/health`
4. Create an issue on GitHub
5. Contact support team

---

**Thank you for using the MLOps PCB Detection System!** 🚀

*This documentation is maintained by [Your Name/Team]*
*Last updated: November 2025*