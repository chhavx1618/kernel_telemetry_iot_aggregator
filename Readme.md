# IoT Kernel Telemetry ML-Based Anomaly Detection System

Machine learning-based anomaly detection for Raspberry Pi clusters using Prometheus metrics.

## Overview

This system uses Isolation Forest and statistical methods to detect anomalies in IoT device telemetry data. It monitors CPU, memory, disk, and network metrics to identify potential security threats like cryptojacking, botnets, and DDoS attacks.

## Features

- Multi-model ensemble anomaly detection (Isolation Forest + statistical analysis)
- Pattern-based threat detection (cryptojacking, botnet activity, memory leaks, DDoS)
- Real-time monitoring with Prometheus integration
- Severity classification (LOW, MEDIUM, HIGH)
- Feature engineering for complex pattern recognition
- Model persistence for deployment

## Requirements

```
numpy
pandas
scikit-learn
joblib
requests
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn joblib requests
```

## Prerequisites

- Prometheus server running (default: http://localhost:9090)
- Node exporter running on each monitored device (default port: 9100)
- At least 1 hour of historical data (1 week recommended for production)

## Configuration

Edit the following variables in the script:

### Training Mode
```python
PROMETHEUS_URL = 'http://localhost:9090'
NODES = [
    '172.27.137.162:9100',
    '172.27.137.180:9100',
    '172.27.137.231:9100'
]
TRAINING_DAYS = 7  # Recommended: 7-30 days
```

### Monitoring Mode
```python
PROMETHEUS_URL = 'http://localhost:9090'
NODES = [
    '172.27.137.162:9100',
    '172.27.137.180:9100',
    '172.27.137.231:9100'
]
CHECK_INTERVAL_SECONDS = 60
```

## Usage

### Step 1: Train Models

Train models on historical data (run once or periodically to retrain):

```bash
python anomaly_detector.py train
```

This will:
- Fetch historical metrics from Prometheus
- Engineer features and create statistical baselines
- Train Isolation Forest models for each node
- Save models to `iot_models_ensemble_detector.pkl`

### Step 2: Start Monitoring

Run real-time monitoring (continuous):

```bash
python anomaly_detector.py monitor
```

This will:
- Load trained models
- Fetch recent metrics every CHECK_INTERVAL_SECONDS
- Detect anomalies using ML and pattern matching
- Log alerts to `alerts.json`
- Print alerts to console

Press Ctrl+C to stop monitoring.

## Output

### Console Output
```
Monitoring Cycle - 2025-11-06 14:30:00
============================================================

Checking 172.27.137.162:9100...
→ HIGH anomaly at 2025-11-06 14:29:45 | score=0.8234 | triggered: ['cpu_usage', 'cpu_network_ratio']
  ALERT: 1 anomalies detected
  THREAT: 1 patterns detected
    - CRYPTOJACKING: Sustained high CPU usage for 10+ minutes
```

### Alert Log (alerts.json)
```json
[
  {
    "node": "172.27.137.162:9100",
    "type": "CRYPTOJACKING",
    "severity": "CRITICAL",
    "description": "Sustained high CPU usage for 10+ minutes",
    "timestamp": "2025-11-06T14:30:00"
  }
]
```

## Monitored Metrics

- **CPU Usage**: Percentage of CPU utilization
- **Memory Usage**: Percentage of memory used
- **Disk Usage**: Percentage of disk space used (root partition)
- **Network In/Out**: Bytes per second transmitted/received
- **Load Average**: System load average (1 minute)

## Threat Detection Patterns

1. **Cryptojacking**: Sustained CPU usage >80% for 10+ minutes
2. **Botnet Activity**: Network traffic >3 standard deviations above baseline
3. **Memory Leak**: Gradual monotonic increase in memory usage
4. **DDoS Pattern**: Abnormally high network connections (requires custom metric)

## Model Parameters

- **contamination**: Expected proportion of anomalies (default: 0.1 = 10%)
  - Lower value = stricter detection, more alerts
  - Higher value = more permissive, fewer alerts

- **n_estimators**: Number of Isolation Forest trees (default: 200)
  - More trees = more stable predictions but slower

## File Structure

```
.
├── anomaly_detector.py              # Main script
├── iot_models_ensemble_detector.pkl # Trained models (generated)
└── alerts.json                      # Alert log (generated)
```

## Troubleshooting

### No data available for node
- Check Prometheus is running: `curl http://localhost:9090/api/v1/status/config`
- Verify node exporter is running: `curl http://NODE_IP:9100/metrics`
- Check node instance format matches Prometheus labels

### Not enough training data
- Increase TRAINING_DAYS (recommended: 7-30 days)
- Minimum 100 samples required, 1000+ recommended

### No models found
- Run training mode first: `python anomaly_detector.py train`
- Check `iot_models_ensemble_detector.pkl` file exists

### Prometheus query timeout
- Reduce TRAINING_DAYS or increase query timeout in PrometheusConnector
- Check Prometheus server load

## Customization

### Add Custom Metrics
Edit `fetch_node_metrics()` in PrometheusConnector class to add PromQL queries.

### Adjust Detection Thresholds
- Cryptojacking: Change `window_minutes` parameter
- Botnet: Change `threshold_multiplier` parameter
- Memory leak: Change `window` parameter

### Add Alert Integrations
Edit `main_monitoring_mode()` to add:
- Slack webhooks
- Email notifications (SMTP)
- Grafana annotations
- PagerDuty alerts

## Performance

- Training time: ~10-30 seconds per node (1 week of data)
- Monitoring cycle: ~2-5 seconds per node
- Memory usage: ~100-200 MB with models loaded
- Disk usage: ~1-10 MB per trained model

## Notes

- Models should be retrained periodically (weekly/monthly) to adapt to changing patterns
- First 24 hours may have false positives as system learns normal behavior
- Adjust contamination parameter based on your environment's baseline noise
- For production, run as systemd service for automatic restart

## License

Use freely for IoT security monitoring and research.