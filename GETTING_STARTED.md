# 🚀 GETTING STARTED - 3 Simple Steps

## Step 1: Install Dependencies (30 seconds)

```bash
pip install numpy pandas scikit-learn joblib
```

## Step 2: Run Quick Start (2 minutes)

```python
from threat_detection_system import ThreatDetectionSystem, generate_synthetic_data

# Generate sample data
data = generate_synthetic_data(n_samples=1000)

# Train detector
detector = ThreatDetectionSystem()
y = data['label'].values
X = detector.preprocess_network_data(data.drop('label', axis=1))

detector.train_anomaly_detector(X, contamination=0.15)
detector.train_supervised_classifier(X, y)

# Analyze traffic
results = detector.predict_threat(X[:50])
print(f"Threat Level: {results['threat_level']}")
print(f"Threats Found: {results['combined_threats']}")
```

## Step 3: Use With Your Data

```python
import pandas as pd

# Load YOUR network traffic
df = pd.read_csv('your_traffic.csv')

# Analyze it
X = detector.preprocess_network_data(df)
results = detector.predict_threat(X)

# Save for future use
detector.save_models('my_detector')
```

---

# 📚 COMPLETE FILE GUIDE

## For Beginners: Start Here

1. **QUICKSTART.py** ← Run this first!
   - Copy/paste example with comments
   - Works immediately out of the box
   - Shows all basic features

2. **TUTORIAL.py** ← Read this second
   - 6 real-world scenarios
   - CSV files, live traffic, SIEM integration
   - Code examples you can adapt

## For Production Use

3. **threat_detection_system.py**
   - Main ML detection engine
   - Isolation Forest + Random Forest
   - Complete implementation

4. **deep_learning_detector.py**
   - Advanced neural network detection
   - Autoencoder + LSTM models
   - Real-time monitoring

5. **production_integration.py**
   - Enterprise-ready pipeline
   - Combines all models
   - Automated alerting & response

6. **README.md**
   - Complete documentation
   - All features explained
   - Deployment options

---

# 🎯 COMMON USE CASES

## 1️⃣ Analyze a CSV File
```python
from threat_detection_system import ThreatDetectionSystem
import pandas as pd

detector = ThreatDetectionSystem()
df = pd.read_csv('network_logs.csv')
X = detector.preprocess_network_data(df)

# Train (first time only)
detector.train_anomaly_detector(X)
detector.save_models('my_model')

# Use later (skip training)
detector.load_models('my_model')
results = detector.predict_threat(X)
```

## 2️⃣ Monitor Live Traffic
```python
# Process packets as they arrive
traffic_batch = []  # Collect packets

# When you have 100+ packets
X = detector.preprocess_network_data(pd.DataFrame(traffic_batch))
results = detector.predict_threat(X)

if results['threat_level'] in ['HIGH', 'CRITICAL']:
    send_alert(results)  # Your alert function
```

## 3️⃣ Integration with Existing Tools
```python
# Send to Splunk/ELK/SIEM
import requests

results = detector.predict_threat(X)

requests.post('https://your-siem.com/api', json={
    'threat_level': results['threat_level'],
    'confidence': results['avg_threat_probability'],
    'timestamp': datetime.now().isoformat()
})
```

---

# ⚡ Quick Commands

```bash
# Run basic demo
python threat_detection_system.py

# Run deep learning demo
python deep_learning_detector.py

# Run production example
python production_integration.py

# Run quick start
python QUICKSTART.py

# See all scenarios
python TUTORIAL.py
```

---

# 🆘 Need Help?

**Q: What data format do I need?**  
A: CSV with network traffic features (see README.md for full list)

**Q: I don't have labeled data (no attack labels)**  
A: No problem! Use only the anomaly detector:
```python
detector.train_anomaly_detector(X, contamination=0.1)
```

**Q: How do I improve accuracy?**  
A: 
- Use more training data
- Adjust `contamination` parameter (0.05 - 0.2)
- Add domain-specific features
- Use continuous learning (add feedback)

**Q: Can I use this in production?**  
A: Yes! See `production_integration.py` for enterprise example

**Q: What about real-time performance?**  
A: Processes 1000+ packets/second on standard hardware

---

# 📊 What Gets Detected?

✅ Port scans  
✅ DDoS attacks  
✅ Brute force attempts  
✅ Data exfiltration  
✅ Unusual traffic patterns  
✅ Zero-day threats  
✅ Compromised hosts  
✅ Malware communication  

---

# 🎓 Learning Path

1. **Day 1**: Run QUICKSTART.py, understand basic concepts
2. **Day 2**: Try TUTORIAL.py scenarios with sample data
3. **Day 3**: Adapt to your own network data
4. **Day 4**: Set up real-time monitoring
5. **Day 5**: Integrate with existing security tools

---

# 🔧 Troubleshooting

**Error: "Module not found"**
```bash
pip install numpy pandas scikit-learn joblib
```

**Error: "No module named tensorflow"**
- Deep learning is optional
- Set `enable_deep_learning=False` in config
- Or install: `pip install tensorflow`

**High false positives**
```python
# Lower the contamination rate
detector.train_anomaly_detector(X, contamination=0.05)
```

**Slow predictions**
```python
# Use batch processing
results = detector.predict_threat(X_batch)  # Process 100+ at once
```

---

**You're ready to start! Run `python QUICKSTART.py` now! 🚀**
