"""
Advanced Threat Detection System using AI/ML
Analyzes network traffic in real-time to identify anomalous behavior and potential threats
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ThreatDetectionSystem:
    """
    Multi-model threat detection system combining supervised and unsupervised learning
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.random_forest = None
        self.feature_names = None
        
    def preprocess_network_data(self, df):
        """
        Preprocess network traffic data for ML models
        
        Args:
            df: DataFrame with network traffic features
            
        Returns:
            Preprocessed feature matrix
        """
        # Select relevant features for threat detection
        feature_columns = [
            'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'num_compromised',
            'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        
        # Extract features that exist in the dataframe
        available_features = [f for f in feature_columns if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def train_anomaly_detector(self, X, contamination=0.1):
        """
        Train unsupervised anomaly detection model (Isolation Forest)
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of anomalies in the dataset
        """
        print("[*] Training Isolation Forest for anomaly detection...")
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)
        print("[✓] Anomaly detector trained successfully")
        
    def train_supervised_classifier(self, X, y):
        """
        Train supervised classification model for known threat types
        
        Args:
            X: Feature matrix
            y: Labels (0 for normal, 1 for attack)
        """
        print("[*] Training Random Forest classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.random_forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.random_forest.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.random_forest.predict(X_test)
        print("\n[✓] Supervised classifier trained successfully")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Attack']))
        
        return X_test, y_test, y_pred
    
    def predict_threat(self, network_record):
        """
        Predict if network traffic is a threat using ensemble approach
        
        Args:
            network_record: Single network traffic record or batch
            
        Returns:
            Dictionary with threat assessment
        """
        # Ensure input is 2D
        if len(network_record.shape) == 1:
            network_record = network_record.reshape(1, -1)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'records_analyzed': len(network_record)
        }
        
        # Anomaly detection (unsupervised)
        if self.isolation_forest:
            anomaly_scores = self.isolation_forest.decision_function(network_record)
            anomaly_predictions = self.isolation_forest.predict(network_record)
            
            results['anomaly_detected'] = int(np.sum(anomaly_predictions == -1))
            results['avg_anomaly_score'] = float(np.mean(anomaly_scores))
        
        # Supervised classification
        if self.random_forest:
            threat_predictions = self.random_forest.predict(network_record)
            threat_probabilities = self.random_forest.predict_proba(network_record)
            
            results['threats_detected'] = int(np.sum(threat_predictions == 1))
            results['avg_threat_probability'] = float(np.mean(threat_probabilities[:, 1]))
        
        # Ensemble decision
        if self.isolation_forest and self.random_forest:
            # Combine both models for final decision
            combined_threats = (anomaly_predictions == -1) | (threat_predictions == 1)
            results['combined_threats'] = int(np.sum(combined_threats))
            results['threat_level'] = self._assess_threat_level(results)
        
        return results
    
    def _assess_threat_level(self, results):
        """Assess overall threat level based on multiple indicators"""
        if results['avg_threat_probability'] > 0.8 or results['avg_anomaly_score'] < -0.5:
            return 'CRITICAL'
        elif results['avg_threat_probability'] > 0.6 or results['avg_anomaly_score'] < -0.3:
            return 'HIGH'
        elif results['avg_threat_probability'] > 0.4 or results['avg_anomaly_score'] < -0.1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_feature_importance(self):
        """Get feature importance from supervised model"""
        if self.random_forest and self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.random_forest.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def save_models(self, prefix='threat_detector'):
        """Save trained models to disk"""
        if self.isolation_forest:
            joblib.dump(self.isolation_forest, f'{prefix}_isolation_forest.pkl')
        if self.random_forest:
            joblib.dump(self.random_forest, f'{prefix}_random_forest.pkl')
        joblib.dump(self.scaler, f'{prefix}_scaler.pkl')
        print(f"[✓] Models saved with prefix: {prefix}")
    
    def load_models(self, prefix='threat_detector'):
        """Load trained models from disk"""
        try:
            self.isolation_forest = joblib.load(f'{prefix}_isolation_forest.pkl')
            self.random_forest = joblib.load(f'{prefix}_random_forest.pkl')
            self.scaler = joblib.load(f'{prefix}_scaler.pkl')
            print("[✓] Models loaded successfully")
        except FileNotFoundError as e:
            print(f"[!] Error loading models: {e}")


def generate_synthetic_data(n_samples=10000, anomaly_ratio=0.1):
    """
    Generate synthetic network traffic data for demonstration
    
    Args:
        n_samples: Number of samples to generate
        anomaly_ratio: Proportion of anomalous/attack samples
    """
    np.random.seed(42)
    
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_attack = n_samples - n_normal
    
    # Normal traffic patterns
    normal_data = {
        'duration': np.random.exponential(2, n_normal),
        'src_bytes': np.random.lognormal(8, 2, n_normal),
        'dst_bytes': np.random.lognormal(7, 2, n_normal),
        'wrong_fragment': np.random.poisson(0.1, n_normal),
        'urgent': np.random.poisson(0.05, n_normal),
        'hot': np.random.poisson(0.5, n_normal),
        'num_failed_logins': np.zeros(n_normal),
        'num_compromised': np.zeros(n_normal),
        'num_root': np.random.poisson(0.1, n_normal),
        'num_file_creations': np.random.poisson(0.2, n_normal),
        'num_shells': np.zeros(n_normal),
        'num_access_files': np.random.poisson(0.3, n_normal),
        'count': np.random.randint(1, 100, n_normal),
        'srv_count': np.random.randint(1, 50, n_normal),
        'serror_rate': np.random.beta(1, 20, n_normal),
        'srv_serror_rate': np.random.beta(1, 20, n_normal),
        'rerror_rate': np.random.beta(1, 20, n_normal),
        'srv_rerror_rate': np.random.beta(1, 20, n_normal),
        'same_srv_rate': np.random.beta(10, 2, n_normal),
        'diff_srv_rate': np.random.beta(2, 10, n_normal),
        'srv_diff_host_rate': np.random.beta(2, 10, n_normal),
        'dst_host_count': np.random.randint(1, 255, n_normal),
        'dst_host_srv_count': np.random.randint(1, 255, n_normal),
        'dst_host_same_srv_rate': np.random.beta(10, 2, n_normal),
        'dst_host_diff_srv_rate': np.random.beta(2, 10, n_normal),
        'dst_host_same_src_port_rate': np.random.beta(10, 2, n_normal),
        'dst_host_srv_diff_host_rate': np.random.beta(2, 10, n_normal),
        'dst_host_serror_rate': np.random.beta(1, 20, n_normal),
        'dst_host_srv_serror_rate': np.random.beta(1, 20, n_normal),
        'dst_host_rerror_rate': np.random.beta(1, 20, n_normal),
        'dst_host_srv_rerror_rate': np.random.beta(1, 20, n_normal),
        'label': np.zeros(n_normal)
    }
    
    # Attack traffic patterns (anomalous)
    attack_data = {
        'duration': np.random.exponential(5, n_attack),  # Longer durations
        'src_bytes': np.random.lognormal(10, 3, n_attack),  # More bytes
        'dst_bytes': np.random.lognormal(9, 3, n_attack),
        'wrong_fragment': np.random.poisson(2, n_attack),  # More fragments
        'urgent': np.random.poisson(1, n_attack),
        'hot': np.random.poisson(3, n_attack),  # More hot indicators
        'num_failed_logins': np.random.poisson(2, n_attack),  # Failed logins
        'num_compromised': np.random.poisson(1, n_attack),  # Compromised
        'num_root': np.random.poisson(1, n_attack),  # Root access attempts
        'num_file_creations': np.random.poisson(3, n_attack),
        'num_shells': np.random.poisson(0.5, n_attack),  # Shell access
        'num_access_files': np.random.poisson(2, n_attack),
        'count': np.random.randint(50, 500, n_attack),  # High connection count
        'srv_count': np.random.randint(1, 10, n_attack),
        'serror_rate': np.random.beta(5, 5, n_attack),  # Higher error rates
        'srv_serror_rate': np.random.beta(5, 5, n_attack),
        'rerror_rate': np.random.beta(5, 5, n_attack),
        'srv_rerror_rate': np.random.beta(5, 5, n_attack),
        'same_srv_rate': np.random.beta(2, 10, n_attack),  # Different pattern
        'diff_srv_rate': np.random.beta(10, 2, n_attack),
        'srv_diff_host_rate': np.random.beta(10, 2, n_attack),
        'dst_host_count': np.random.randint(100, 255, n_attack),  # Scanning
        'dst_host_srv_count': np.random.randint(1, 50, n_attack),
        'dst_host_same_srv_rate': np.random.beta(2, 10, n_attack),
        'dst_host_diff_srv_rate': np.random.beta(10, 2, n_attack),
        'dst_host_same_src_port_rate': np.random.beta(2, 10, n_attack),
        'dst_host_srv_diff_host_rate': np.random.beta(10, 2, n_attack),
        'dst_host_serror_rate': np.random.beta(5, 5, n_attack),
        'dst_host_srv_serror_rate': np.random.beta(5, 5, n_attack),
        'dst_host_rerror_rate': np.random.beta(5, 5, n_attack),
        'dst_host_srv_rerror_rate': np.random.beta(5, 5, n_attack),
        'label': np.ones(n_attack)
    }
    
    # Combine and shuffle
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Demonstration of the threat detection system"""
    
    print("="*70)
    print("Advanced Threat Detection System - Training & Demonstration")
    print("="*70)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic network traffic data...")
    df = generate_synthetic_data(n_samples=10000, anomaly_ratio=0.15)
    print(f"    Generated {len(df)} network traffic records")
    print(f"    Normal traffic: {sum(df['label'] == 0)}")
    print(f"    Attack traffic: {sum(df['label'] == 1)}")
    
    # Initialize system
    print("\n[2] Initializing threat detection system...")
    tds = ThreatDetectionSystem()
    
    # Preprocess data
    print("\n[3] Preprocessing network data...")
    y = df['label'].values
    df_features = df.drop('label', axis=1)
    X = tds.preprocess_network_data(df_features)
    print(f"    Feature matrix shape: {X.shape}")
    
    # Train models
    print("\n[4] Training ML models...")
    print("-" * 70)
    
    # Train anomaly detector (unsupervised)
    tds.train_anomaly_detector(X, contamination=0.15)
    
    # Train supervised classifier
    X_test, y_test, y_pred = tds.train_supervised_classifier(X, y)
    
    # Feature importance
    print("\n[5] Top 10 Most Important Features:")
    print("-" * 70)
    feature_importance = tds.get_feature_importance()
    print(feature_importance.head(10).to_string(index=False))
    
    # Real-time detection demonstration
    print("\n[6] Real-time Threat Detection Demonstration:")
    print("-" * 70)
    
    # Test on new samples
    test_indices = np.random.choice(len(X), 100, replace=False)
    test_batch = X[test_indices]
    
    results = tds.predict_threat(test_batch)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Timestamp: {results['timestamp']}")
    print(f"   Records Analyzed: {results['records_analyzed']}")
    print(f"   Anomalies Detected: {results['anomaly_detected']}")
    print(f"   Threats Detected (Supervised): {results['threats_detected']}")
    print(f"   Combined Threats: {results['combined_threats']}")
    print(f"   Average Threat Probability: {results['avg_threat_probability']:.3f}")
    print(f"   Average Anomaly Score: {results['avg_anomaly_score']:.3f}")
    print(f"   🚨 Threat Level: {results['threat_level']}")
    
    # Save models
    print("\n[7] Saving trained models...")
    tds.save_models()
    
    print("\n" + "="*70)
    print("✓ Threat Detection System Ready for Deployment")
    print("="*70)
    print("\nThe system can now be used to:")
    print("  • Monitor network traffic in real-time")
    print("  • Detect zero-day threats using anomaly detection")
    print("  • Classify known attack patterns")
    print("  • Provide threat level assessments")
    print("  • Generate alerts for security teams")


if __name__ == "__main__":
    main()
