"""
Generate comprehensive training data for ML models
This script creates detailed datasets for network analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_traffic_data(hours=168):  # One week of data
    """Generate realistic traffic load data"""
    data = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    for i in range(hours):
        timestamp = start_time + timedelta(hours=i)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Base traffic varies by time
        if 9 <= hour <= 17:  # Business hours
            base_load = 70
            variance = 20
        elif 18 <= hour <= 23:  # Evening
            base_load = 80
            variance = 25
        else:  # Night/early morning
            base_load = 30
            variance = 15
        
        # Weekend pattern is different
        if is_weekend:
            base_load = base_load * 0.7
        
        load = base_load + np.random.normal(0, variance)
        load = max(0, load)  # Ensure non-negative
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'traffic_load': round(load, 2)
        })
    
    return pd.DataFrame(data)

def generate_qos_data(n_samples=1000):
    """Generate QoS metrics data"""
    data = []
    
    for i in range(n_samples):
        # Random network conditions
        load = np.random.uniform(0, 100)
        distance = np.random.uniform(0, 5000)
        rssi = np.random.uniform(-100, -60)
        num_ues = np.random.randint(1, 50)
        is_macro = np.random.choice([0, 1])
        
        # Calculate latency (ms)
        base_latency = 10
        load_penalty = load * 0.3
        distance_penalty = distance * 0.0008
        rssi_penalty = (rssi + 80) * 0.5  # Better RSSI = lower latency
        latency = base_latency + load_penalty + distance_penalty + rssi_penalty
        latency += np.random.normal(0, 5)  # Add noise
        latency = max(5, latency)
        
        # Calculate throughput (Mbps)
        base_throughput = 100
        load_reduction = load * 0.4
        distance_reduction = distance * 0.008
        rssi_impact = (rssi + 80) * 0.3
        throughput = base_throughput - load_reduction - distance_reduction - rssi_impact
        throughput += np.random.normal(0, 10)
        throughput = max(10, throughput)
        
        # Packet loss (%)
        packet_loss = max(0, load * 0.01 + distance * 0.0001 + np.random.normal(0, 0.5))
        packet_loss = min(20, packet_loss)
        
        data.append({
            'load': round(load, 2),
            'distance': round(distance, 2),
            'rssi': round(rssi, 2),
            'num_ues': num_ues,
            'is_macro': is_macro,
            'latency': round(latency, 2),
            'throughput': round(throughput, 2),
            'packet_loss': round(packet_loss, 2)
        })
    
    return pd.DataFrame(data)

def generate_handover_data(n_samples=1000):
    """Generate handover decision data"""
    data = []
    
    for i in range(n_samples):
        # Current cell metrics
        current_rssi = np.random.uniform(-100, -60)
        current_load = np.random.uniform(0, 100)
        current_distance = np.random.uniform(0, 5000)
        current_latency = np.random.uniform(10, 100)
        
        # Target cell metrics
        target_rssi = np.random.uniform(-100, -60)
        target_load = np.random.uniform(0, 100)
        target_distance = np.random.uniform(0, 5000)
        
        # Determine if handover should occur
        rssi_diff = target_rssi - current_rssi
        load_diff = current_load - target_load
        distance_improvement = current_distance - target_distance
        
        # Handover decision logic
        should_handover = 0
        score = 0
        
        if rssi_diff > 5:  # Better signal
            score += 2
        elif rssi_diff > 10:
            score += 3
        
        if load_diff > 20:  # Much lower load
            score += 2
        
        if distance_improvement > 1000:  # Significant distance improvement
            score += 1
        
        # Random handovers based on mobility
        mobility = np.random.random()
        if mobility > 0.7:
            score += 1
        
        should_handover = 1 if score >= 3 else 0
        
        # Random noise
        if np.random.random() < 0.1:  # 10% false positives
            should_handover = 1 - should_handover
        
        data.append({
            'current_rssi': round(current_rssi, 2),
            'current_load': round(current_load, 2),
            'current_distance': round(current_distance, 2),
            'current_latency': round(current_latency, 2),
            'target_rssi': round(target_rssi, 2),
            'target_load': round(target_load, 2),
            'target_distance': round(target_distance, 2),
            'rssi_improvement': round(rssi_diff, 2),
            'load_improvement': round(load_diff, 2),
            'distance_improvement': round(distance_improvement, 2),
            'should_handover': should_handover
        })
    
    return pd.DataFrame(data)

def generate_anomaly_data(n_samples=1000):
    """Generate anomaly detection data"""
    data = []
    
    for i in range(n_samples):
        normal_load = np.random.uniform(20, 60)
        normal_rssi = np.random.uniform(-80, -65)
        
        # Occasionally add anomalies
        if np.random.random() < 0.1:  # 10% anomalies
            load = np.random.uniform(80, 120)  # Overload
            anomaly = 1
        elif np.random.random() < 0.05:  # 5% anomalies
            load = np.random.uniform(0, 5)  # Underload/unusual pattern
            anomaly = 1
        else:
            load = normal_load
            anomaly = 0
        
        rssi = normal_rssi
        if anomaly:
            # Anomalies often have poor signal quality
            rssi = np.random.uniform(-95, -85)
        
        data.append({
            'timestamp': i,
            'load': round(load, 2),
            'rssi': round(rssi, 2),
            'num_connections': int(load / 10),
            'is_anomaly': anomaly
        })
    
    return pd.DataFrame(data)

def main():
    """Generate all training datasets"""
    print("Generating training datasets...")
    
    print("1. Generating traffic load data...")
    traffic_df = generate_traffic_data(hours=720)  # 30 days
    traffic_df.to_csv('traffic_training_data.csv', index=False)
    print(f"   Generated {len(traffic_df)} traffic samples")
    
    print("2. Generating QoS prediction data...")
    qos_df = generate_qos_data(n_samples=5000)
    qos_df.to_csv('qos_training_data.csv', index=False)
    print(f"   Generated {len(qos_df)} QoS samples")
    
    print("3. Generating handover decision data...")
    handover_df = generate_handover_data(n_samples=3000)
    handover_df.to_csv('handover_training_data.csv', index=False)
    print(f"   Generated {len(handover_df)} handover samples")
    
    print("4. Generating anomaly detection data...")
    anomaly_df = generate_anomaly_data(n_samples=2000)
    anomaly_df.to_csv('anomaly_training_data.csv', index=False)
    print(f"   Generated {len(anomaly_df)} anomaly samples")
    
    print("\nAll training datasets generated successfully!")
    print("\nFiles created:")
    print("  - traffic_training_data.csv")
    print("  - qos_training_data.csv")
    print("  - handover_training_data.csv")
    print("  - anomaly_training_data.csv")

if __name__ == '__main__':
    main()

