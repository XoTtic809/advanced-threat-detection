"""
QUICK START GUIDE - Advanced Threat Detection System
=====================================================

This guide will help you get started in 5 minutes!
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================

"""
Open your terminal and run:

pip install numpy pandas scikit-learn joblib

Optional (for deep learning features):
pip install tensorflow keras
"""

# ============================================================================
# STEP 2: BASIC USAGE - DETECT THREATS IN YOUR DATA
# ============================================================================

from threat_detection_system import ThreatDetectionSystem, generate_synthetic_data
import pandas as pd

# Option A: Use with your own data
# ---------------------------------
# Load your network traffic CSV file
# Your CSV should have columns like: duration, src_bytes, dst_bytes, etc.
# See README.md for full list of features

# my_data = pd.read_csv('my_network_traffic.csv')


# Option B: Start with example data (recommended for testing)
# ------------------------------------------------------------
print("Generating example network traffic data...")
data = generate_synthetic_data(n_samples=1000, anomaly_ratio=0.15)
print(f"✓ Generated {len(data)} traffic records")

# ============================================================================
# STEP 3: TRAIN THE DETECTOR
# ============================================================================

print("\n[1] Training threat detector...")

# Initialize the system
detector = ThreatDetectionSystem()

# Separate features from labels (if you have labels)
y = data['label'].values  # 0 = normal, 1 = attack
X = detector.preprocess_network_data(data.drop('label', axis=1))

# Train the models (this takes ~10 seconds)
detector.train_anomaly_detector(X, contamination=0.15)
detector.train_supervised_classifier(X, y)

print("✓ Training complete!")

# ============================================================================
# STEP 4: DETECT THREATS IN NEW TRAFFIC
# ============================================================================

print("\n[2] Analyzing new network traffic...")

# Get some new traffic to analyze (in production, this would be live traffic)
new_traffic = X[:50]  # Analyze first 50 records

# Detect threats
results = detector.predict_threat(new_traffic)

# Print results
print(f"\n📊 THREAT ANALYSIS RESULTS:")
print(f"   Records Analyzed: {results['records_analyzed']}")
print(f"   Threats Detected: {results['combined_threats']}")
print(f"   Threat Probability: {results['avg_threat_probability']:.1%}")
print(f"   🚨 Threat Level: {results['threat_level']}")

# ============================================================================
# STEP 5: SAVE YOUR TRAINED MODELS
# ============================================================================

print("\n[3] Saving models for future use...")
detector.save_models('my_threat_detector')
print("✓ Models saved! You can load them later without retraining.")

# ============================================================================
# STEP 6: USE SAVED MODELS (Skip training next time)
# ============================================================================

print("\n[4] Loading saved models...")

# Create new detector instance
new_detector = ThreatDetectionSystem()

# Load previously trained models
new_detector.load_models('my_threat_detector')

# Use immediately for predictions
quick_results = new_detector.predict_threat(new_traffic[:10])
print(f"✓ Quick analysis: {quick_results['combined_threats']} threats in 10 records")

# ============================================================================
# BONUS: SEE WHICH FEATURES ARE MOST IMPORTANT
# ============================================================================

print("\n[5] Top threat indicators:")
importance = detector.get_feature_importance()
print(importance.head(5).to_string(index=False))

print("\n" + "="*60)
print("✓ QUICK START COMPLETE!")
print("="*60)
print("\nWhat's next?")
print("  → Check README.md for advanced features")
print("  → Run 'python threat_detection_system.py' for full demo")
print("  → Run 'python production_integration.py' for enterprise example")
