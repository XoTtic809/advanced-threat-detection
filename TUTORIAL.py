"""
PRACTICAL TUTORIAL: Using Threat Detection with YOUR Data
==========================================================

This tutorial shows you how to use the system with your own network traffic data.
"""

# ============================================================================
# SCENARIO 1: I HAVE A CSV FILE WITH NETWORK TRAFFIC
# ============================================================================

"""
Your CSV might look like this:

duration,src_bytes,dst_bytes,protocol,service,...
0.5,1024,512,tcp,http,...
1.2,2048,1024,tcp,ftp,...
0.3,512,256,udp,dns,...

STEP-BY-STEP:
"""

import pandas as pd
from threat_detection_system import ThreatDetectionSystem

def analyze_my_csv_file():
    """Analyze network traffic from a CSV file"""
    
    # 1. Load your CSV
    df = pd.read_csv('your_network_traffic.csv')
    
    # 2. Initialize detector
    detector = ThreatDetectionSystem()
    
    # 3. If you have labels (known attacks), train supervised model
    if 'label' in df.columns:
        y = df['label'].values
        X = detector.preprocess_network_data(df.drop('label', axis=1))
        
        detector.train_anomaly_detector(X, contamination=0.1)
        detector.train_supervised_classifier(X, y)
        detector.save_models('my_trained_model')
    
    # 4. If you DON'T have labels, use unsupervised only
    else:
        X = detector.preprocess_network_data(df)
        detector.train_anomaly_detector(X, contamination=0.1)
        detector.save_models('my_trained_model')
    
    # 5. Analyze for threats
    results = detector.predict_threat(X)
    
    # 6. Print summary
    print(f"Analyzed {results['records_analyzed']} records")
    print(f"Found {results['combined_threats']} potential threats")
    print(f"Threat Level: {results['threat_level']}")
    
    return results


# ============================================================================
# SCENARIO 2: I'M CAPTURING LIVE NETWORK TRAFFIC
# ============================================================================

"""
If you're using Wireshark, tcpdump, or similar tools:

1. Export to CSV from Wireshark
2. Or use scapy to capture in Python
"""

def monitor_live_traffic():
    """Monitor live network traffic"""
    
    from scapy.all import sniff, IP, TCP
    import numpy as np
    
    # Load pre-trained model
    detector = ThreatDetectionSystem()
    detector.load_models('my_trained_model')
    
    traffic_buffer = []
    
    def process_packet(packet):
        """Process each captured packet"""
        
        if packet.haslayer(IP) and packet.haslayer(TCP):
            # Extract basic features
            features = {
                'duration': 0,  # Would calculate from flow
                'src_bytes': len(packet),
                'dst_bytes': len(packet),
                # ... add more features based on your needs
            }
            
            traffic_buffer.append(features)
            
            # Analyze in batches of 100
            if len(traffic_buffer) >= 100:
                df = pd.DataFrame(traffic_buffer)
                X = detector.preprocess_network_data(df)
                results = detector.predict_threat(X)
                
                if results['threat_level'] in ['HIGH', 'CRITICAL']:
                    print(f"⚠️  ALERT: {results['threat_level']} threat detected!")
                    print(f"   Probability: {results['avg_threat_probability']:.1%}")
                
                traffic_buffer.clear()
    
    # Start capturing (requires root/admin privileges)
    # sniff(prn=process_packet, store=0, count=1000)
    
    print("Live monitoring example (commented out - requires scapy)")


# ============================================================================
# SCENARIO 3: I WANT TO ANALYZE A SPECIFIC IP ADDRESS
# ============================================================================

def analyze_specific_ip(df, target_ip):
    """Analyze traffic from/to a specific IP"""
    
    # Filter data for specific IP
    traffic = df[(df['src_ip'] == target_ip) | (df['dst_ip'] == target_ip)]
    
    print(f"Analyzing {len(traffic)} packets for IP {target_ip}")
    
    # Initialize and load model
    detector = ThreatDetectionSystem()
    detector.load_models('my_trained_model')
    
    # Prepare data
    X = detector.preprocess_network_data(traffic.drop(['src_ip', 'dst_ip'], axis=1))
    
    # Analyze
    results = detector.predict_threat(X)
    
    print(f"\nResults for {target_ip}:")
    print(f"  Threat Level: {results['threat_level']}")
    print(f"  Confidence: {results['avg_threat_probability']:.1%}")
    
    return results


# ============================================================================
# SCENARIO 4: REAL-TIME MONITORING WITH ALERTS
# ============================================================================

def setup_real_time_monitoring():
    """Set up continuous monitoring with email/Slack alerts"""
    
    import time
    from datetime import datetime
    
    # Load trained model
    detector = ThreatDetectionSystem()
    detector.load_models('my_trained_model')
    
    def send_alert(results):
        """Send alert via email or Slack"""
        # Email example
        import smtplib
        from email.message import EmailMessage
        
        if results['threat_level'] in ['HIGH', 'CRITICAL']:
            msg = EmailMessage()
            msg['Subject'] = f"🚨 {results['threat_level']} Threat Detected"
            msg['From'] = 'security@company.com'
            msg['To'] = 'soc-team@company.com'
            msg.set_content(f"""
            Threat Detection Alert
            
            Timestamp: {datetime.now()}
            Threat Level: {results['threat_level']}
            Confidence: {results['avg_threat_probability']:.1%}
            Threats Detected: {results['combined_threats']}
            
            Please investigate immediately.
            """)
            
            # Send email (configure SMTP settings)
            # smtp = smtplib.SMTP('smtp.company.com', 587)
            # smtp.send_message(msg)
            
            print(f"📧 Alert sent: {results['threat_level']} threat")
    
    def monitor_loop():
        """Continuous monitoring loop"""
        
        while True:
            # Read latest network traffic (from file, database, or live capture)
            # df = pd.read_csv('latest_traffic.csv')
            # X = detector.preprocess_network_data(df)
            # results = detector.predict_threat(X)
            
            # send_alert(results)
            
            # Sleep for 60 seconds before next check
            time.sleep(60)
    
    print("Real-time monitoring setup example")
    # monitor_loop()  # Uncomment to run continuously


# ============================================================================
# SCENARIO 5: BATCH ANALYSIS OF HISTORICAL DATA
# ============================================================================

def analyze_historical_logs():
    """Analyze historical network logs to find past attacks"""
    
    detector = ThreatDetectionSystem()
    detector.load_models('my_trained_model')
    
    # Load historical data (could be multiple files)
    all_results = []
    
    for log_file in ['day1.csv', 'day2.csv', 'day3.csv']:
        print(f"\nAnalyzing {log_file}...")
        
        # df = pd.read_csv(log_file)
        # X = detector.preprocess_network_data(df)
        # results = detector.predict_threat(X)
        
        # all_results.append({
        #     'file': log_file,
        #     'threats': results['combined_threats'],
        #     'level': results['threat_level']
        # })
    
    # Generate report
    # report_df = pd.DataFrame(all_results)
    # report_df.to_csv('threat_analysis_report.csv', index=False)
    
    print("Historical analysis example")


# ============================================================================
# SCENARIO 6: INTEGRATING WITH YOUR EXISTING SECURITY STACK
# ============================================================================

def integrate_with_siem():
    """Send detections to SIEM (Splunk, ELK, etc.)"""
    
    import requests
    import json
    
    detector = ThreatDetectionSystem()
    detector.load_models('my_trained_model')
    
    def send_to_splunk(results):
        """Send to Splunk HEC (HTTP Event Collector)"""
        
        splunk_url = "https://splunk.company.com:8088/services/collector"
        splunk_token = "YOUR-HEC-TOKEN"
        
        event = {
            "sourcetype": "threat_detection",
            "event": {
                "timestamp": datetime.now().isoformat(),
                "threat_level": results['threat_level'],
                "confidence": results['avg_threat_probability'],
                "threats_detected": results['combined_threats'],
                "anomaly_score": results['avg_anomaly_score']
            }
        }
        
        headers = {
            "Authorization": f"Splunk {splunk_token}",
            "Content-Type": "application/json"
        }
        
        # response = requests.post(splunk_url, 
        #                         headers=headers, 
        #                         data=json.dumps(event))
        
        print("SIEM integration example")
    
    # Example: Analyze and send to SIEM
    # df = pd.read_csv('network_traffic.csv')
    # X = detector.preprocess_network_data(df)
    # results = detector.predict_threat(X)
    # send_to_splunk(results)


# ============================================================================
# RUNNING THE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PRACTICAL TUTORIAL - Use Cases")
    print("="*70)
    
    print("\n1. CSV File Analysis")
    print("   → Load your data: pd.read_csv('your_file.csv')")
    print("   → Preprocess: detector.preprocess_network_data(df)")
    print("   → Analyze: detector.predict_threat(X)")
    
    print("\n2. Live Traffic Monitoring")
    print("   → Use scapy to capture packets")
    print("   → Process in real-time batches")
    print("   → Generate alerts for threats")
    
    print("\n3. Specific IP Analysis")
    print("   → Filter data by IP address")
    print("   → Analyze behavioral patterns")
    print("   → Identify compromised hosts")
    
    print("\n4. Real-time Alerts")
    print("   → Set up email/Slack notifications")
    print("   → Configure alert thresholds")
    print("   → Continuous monitoring loop")
    
    print("\n5. Historical Analysis")
    print("   → Batch process old logs")
    print("   → Identify past breaches")
    print("   → Generate reports")
    
    print("\n6. SIEM Integration")
    print("   → Send to Splunk/ELK/QRadar")
    print("   → Enrich existing security data")
    print("   → Automated response workflows")
    
    print("\n" + "="*70)
    print("Choose the scenario that fits your needs!")
    print("="*70)
    
    # Run the simple demo from earlier
    print("\nRunning QUICKSTART demo...\n")
    from threat_detection_system import generate_synthetic_data
    
    data = generate_synthetic_data(n_samples=500)
    detector = ThreatDetectionSystem()
    
    y = data['label'].values
    X = detector.preprocess_network_data(data.drop('label', axis=1))
    
    detector.train_anomaly_detector(X, contamination=0.15)
    detector.train_supervised_classifier(X, y)
    
    results = detector.predict_threat(X[:100])
    
    print(f"\n✓ Analyzed 100 records:")
    print(f"  Threats: {results['combined_threats']}")
    print(f"  Level: {results['threat_level']}")
