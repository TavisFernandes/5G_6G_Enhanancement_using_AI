"""
Analyze and visualize network data from simulation
Generates plots and statistics for network performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_qos_metrics():
    """Analyze QoS metrics"""
    try:
        df = pd.read_csv('qos_metrics.csv')
        
        print("\n" + "="*60)
        print("QoS Metrics Analysis")
        print("="*60)
        
        print(f"\nTotal samples: {len(df)}")
        print(f"\nLatency Statistics (ms):")
        print(f"  Mean: {df['latency'].mean():.2f}")
        print(f"  Median: {df['latency'].median():.2f}")
        print(f"  Min: {df['latency'].min():.2f}")
        print(f"  Max: {df['latency'].max():.2f}")
        print(f"  Std Dev: {df['latency'].std():.2f}")
        
        print(f"\nThroughput Statistics (Mbps):")
        print(f"  Mean: {df['throughput'].mean():.2f}")
        print(f"  Median: {df['throughput'].median():.2f}")
        print(f"  Min: {df['throughput'].min():.2f}")
        print(f"  Max: {df['throughput'].max():.2f}")
        print(f"  Std Dev: {df['throughput'].std():.2f}")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Latency distribution
        axes[0, 0].hist(df['latency'], bins=50, color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Latency (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Latency Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Throughput distribution
        axes[0, 1].hist(df['throughput'], bins=50, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Throughput (Mbps)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Throughput Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Latency vs Distance
        axes[1, 0].scatter(df['distance'], df['latency'], alpha=0.3, s=10)
        axes[1, 0].set_xlabel('Distance (m)')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('Latency vs Distance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Throughput vs Distance
        axes[1, 1].scatter(df['distance'], df['throughput'], alpha=0.3, s=10, color='green')
        axes[1, 1].set_xlabel('Distance (m)')
        axes[1, 1].set_ylabel('Throughput (Mbps)')
        axes[1, 1].set_title('Throughput vs Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qos_analysis.png', dpi=150)
        print("\nSaved: qos_analysis.png")
        
    except FileNotFoundError:
        print("QoS metrics file not found. Run simulation first.")

def analyze_traffic_prediction():
    """Analyze traffic predictions"""
    try:
        df = pd.read_csv('prediction_results.csv')
        
        print("\n" + "="*60)
        print("Traffic Prediction Analysis")
        print("="*60)
        
        print(f"\nTotal predictions: {len(df)}")
        
        # Calculate prediction error
        df['prediction_error'] = abs(df['predicted_load'] - df['actual_load'])
        
        print(f"\nPrediction Error Statistics:")
        print(f"  Mean Error: {df['prediction_error'].mean():.2f}")
        print(f"  Median Error: {df['prediction_error'].median():.2f}")
        print(f"  Max Error: {df['prediction_error'].max():.2f}")
        
        # Calculate accuracy (within 10% of actual)
        accuracy = (df['prediction_error'] < df['actual_load'] * 0.1).sum() / len(df) * 100
        print(f"  Accuracy (within 10%): {accuracy:.2f}%")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df['time'], df['predicted_load'], label='Predicted Load', alpha=0.7)
        ax.plot(df['time'], df['actual_load'], label='Actual Load', alpha=0.7)
        ax.fill_between(df['time'], df['predicted_load'], df['actual_load'], alpha=0.2, label='Error')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Traffic Load')
        ax.set_title('Traffic Load Prediction vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('traffic_prediction_analysis.png', dpi=150)
        print("\nSaved: traffic_prediction_analysis.png")
        
    except FileNotFoundError:
        print("Prediction results file not found. Run simulation first.")

def analyze_anomalies():
    """Analyze detected anomalies"""
    try:
        df = pd.read_csv('anomaly_log.csv')
        
        print("\n" + "="*60)
        print("Anomaly Detection Analysis")
        print("="*60)
        
        print(f"\nTotal anomalies detected: {len(df)}")
        
        if len(df) > 0:
            print(f"\nStation Types:")
            print(df['station_type'].value_counts())
            
            print(f"\nLoad Statistics for Anomalies:")
            print(f"  Mean: {df['load'].mean():.2f}")
            print(f"  Min: {df['load'].min():.2f}")
            print(f"  Max: {df['load'].max():.2f}")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.scatter(df['time'], df['load'], c='red', alpha=0.6, s=50)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Load')
            ax.set_title('Detected Network Anomalies')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('anomaly_analysis.png', dpi=150)
            print("\nSaved: anomaly_analysis.png")
        else:
            print("\nNo anomalies detected.")
        
    except FileNotFoundError:
        print("Anomaly log file not found. Run simulation first.")

def analyze_network_topology():
    """Analyze network topology and base station utilization"""
    try:
        df = pd.read_csv('network_data.csv')
        
        print("\n" + "="*60)
        print("Network Topology Analysis")
        print("="*60)
        
        print(f"\nTotal base stations: {len(df['station_id'].unique())}")
        print(f"  Macro cells: {(df['station_type'] == 'macro').sum()}")
        print(f"  Small cells: {(df['station_type'] == 'small').sum()}")
        
        print(f"\nLoad Statistics:")
        print(f"  Mean: {df['load'].mean():.2f}")
        print(f"  Max: {df['load'].max():.2f}")
        print(f"  Min: {df['load'].min():.2f}")
        
        print(f"\nConnection Statistics:")
        print(f"  Mean connections per station: {df['connected_ues'].mean():.2f}")
        print(f"  Max connections: {df['connected_ues'].max()}")
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Load distribution by station type
        macro_loads = df[df['station_type'] == 'macro']['load']
        small_loads = df[df['station_type'] == 'small']['load']
        
        axes[0].hist(macro_loads, bins=20, alpha=0.7, label='Macro', color='blue')
        axes[0].hist(small_loads, bins=20, alpha=0.7, label='Small', color='red')
        axes[0].set_xlabel('Load Unit')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Load Distribution by Cell Type')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Station positions
        ax = axes[1]
        macro_stations = df[df['station_type'] == 'macro']
        small_stations = df[df['station_type'] == 'small']
        
        ax.scatter(macro_stations['x'], macro_stations['y'], c='blue', s=300, marker='^', label='Macro', alpha=0.7)
        ax.scatter(small_stations['x'], small_stations['y'], c='red', s=100, marker='^', label='Small', alpha=0.7)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Network Topology')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('network_topology_analysis.png', dpi=150)
        print("\nSaved: network_topology_analysis.png")
        
    except FileNotFoundError:
        print("Network data file not found. Run simulation first.")

def main():
    """Run all analysis"""
    print("Analyzing Network Simulation Data")
    print("="*60)
    
    analyze_network_topology()
    analyze_qos_metrics()
    analyze_traffic_prediction()
    analyze_anomalies()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()

