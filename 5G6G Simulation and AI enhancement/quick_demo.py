"""
Quick demonstration of 5G network simulation features
Runs a short simulation for testing purposes
"""

import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the simulation module
import importlib.util
spec = importlib.util.spec_from_file_location("sim_module", "5g_network_simulation.py")
sim_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_module)
NetworkSimulator = sim_module.NetworkSimulator

def run_quick_demo():
    """Run a short demo of the network simulation"""
    print("="*60)
    print("5G/6G Network Simulation - Quick Demo")
    print("="*60)
    
    simulator = NetworkSimulator()
    
    print("\nRunning 50 time steps...")
    for i in range(50):
        simulator.time_step = i
        simulator.update_ue_positions()
        simulator.update_connections()
        
        # Update base station load
        for bs in simulator.base_stations:
            bs.current_load = len(bs.connected_ues) * 10
        
        # Traffic prediction
        predicted_load, actual_load = simulator.predict_traffic_load()
        simulator.prediction_log.append({
            'time': i,
            'predicted_load': predicted_load,
            'actual_load': actual_load
        })
        
        # Anomaly detection
        anomalies = simulator.detect_anomalies()
        for anomaly in anomalies:
            simulator.anomaly_log.append(anomaly)
        
        # Log QoS data
        for ue in simulator.ues:
            if ue.connected_bs:
                latency, throughput = simulator.predict_qos(ue)
                ue.latency = latency
                ue.throughput = throughput
                
                simulator.qos_log.append({
                    'time': i,
                    'ue_id': ue.ue_id,
                    'bs_id': ue.connected_bs.station_id,
                    'latency': latency,
                    'throughput': throughput,
                    'distance': np.sqrt((ue.connected_bs.x - ue.x)**2 + (ue.connected_bs.y - ue.y)**2)
                })
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/50 completed")
    
    # Save data
    simulator.save_csv_data()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    
    # Print summary statistics
    print(f"\nNetwork Summary:")
    print(f"  Total Base Stations: {len(simulator.base_stations)}")
    print(f"  Total UEs: {len(simulator.ues)}")
    print(f"  Total Handovers: {sum(ue.handover_count for ue in simulator.ues)}")
    
    if len(simulator.qos_log) > 0:
        avg_latency = np.mean([item['latency'] for item in simulator.qos_log])
        avg_throughput = np.mean([item['throughput'] for item in simulator.qos_log])
        print(f"\nQoS Metrics (avg):")
        print(f"  Average Latency: {avg_latency:.2f} ms")
        print(f"  Average Throughput: {avg_throughput:.2f} Mbps")
    
    if len(simulator.prediction_log) > 0:
        recent_pred = simulator.prediction_log[-10:]
        avg_error = np.mean([abs(item['predicted_load'] - item['actual_load']) 
                            for item in recent_pred])
        print(f"\nTraffic Prediction:")
        print(f"  Average Prediction Error: {avg_error:.2f}")
    
    if len(simulator.anomaly_log) > 0:
        print(f"\nAnomaly Detection:")
        print(f"  Total Anomalies Detected: {len(simulator.anomaly_log)}")
    
    print(f"\nFiles generated:")
    print(f"  - network_data.csv")
    print(f"  - ue_data.csv")
    print(f"  - qos_metrics.csv")
    print(f"  - prediction_results.csv")
    print(f"  - anomaly_log.csv")

if __name__ == '__main__':
    run_quick_demo()

