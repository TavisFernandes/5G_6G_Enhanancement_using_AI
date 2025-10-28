"""
5G/6G Cellular Network Simulation with AI/ML Enhancements
Includes traffic prediction, AI handover, anomaly detection, and QoS prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BaseStation:
    """Represents a base station in the network"""
    def __init__(self, x, y, cell_type='macro', station_id=0, capacity=1000):
        self.x = x
        self.y = y
        self.cell_type = cell_type  # 'macro' or 'small'
        self.station_id = station_id
        self.capacity = capacity  # Mbps
        self.connected_ues = []
        self.current_load = 0
        self.coverage_radius = 5000 if cell_type == 'macro' else 500  # meters
        
    def calculate_path_loss(self, ue_x, ue_y):
        """Calculate path loss based on distance (simplified Friis equation)"""
        distance = np.sqrt((self.x - ue_x)**2 + (self.y - ue_y)**2)
        # Simplified path loss model: PL = 32.4 + 20*log10(f) + 20*log10(d)
        # Using 5G frequency ~3.5 GHz
        return 32.4 + 20 * np.log10(3.5) + 20 * np.log10(distance/1000)
    
    def get_rssi(self, ue_x, ue_y):
        """Calculate received signal strength indicator"""
        path_loss = self.calculate_path_loss(ue_x, ue_y)
        tx_power = 46  # dBm for macro, 30 for small cells
        return tx_power - path_loss

class UserEquipment:
    """Represents user equipment in the network"""
    def __init__(self, ue_id, x, y, velocity=0, direction=0):
        self.ue_id = ue_id
        self.x = x
        self.y = y
        self.velocity = velocity  # m/s
        self.direction = direction  # radians
        self.connected_bs = None
        self.data_rate = 0
        self.latency = 0
        self.throughput = 0
        self.handover_count = 0

class NetworkSimulator:
    """Main network simulator with AI/ML enhancements"""
    
    def __init__(self):
        self.base_stations = []
        self.ues = []
        self.time_step = 0
        self.data_log = []
        self.qos_log = []
        self.prediction_log = []
        self.anomaly_log = []
        
        # ML Models
        self.traffic_predictor = None
        self.handover_predictor = None
        self.qos_predictor = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # Training data
        self.traffic_data = []
        self.handover_data = []
        self.qos_data = []
        
        self.setup_network()
        self.prepare_ml_models()
    
    def setup_network(self):
        """Initialize the network topology"""
        # Create macro cells (wide coverage)
        macro_positions = [
            (0, 0), (10000, 0), (20000, 0),
            (0, 10000), (10000, 10000), (20000, 10000),
            (0, 20000), (10000, 20000), (20000, 20000)
        ]
        
        for i, (x, y) in enumerate(macro_positions):
            self.base_stations.append(BaseStation(x, y, 'macro', i, capacity=1000))
        
        # Create small cells (high-capacity hotspots)
        small_positions = [
            (5000, 5000), (15000, 5000), (5000, 15000), (15000, 15000),
            (2500, 2500), (7500, 7500), (17500, 12500), (12500, 17500)
        ]
        
        for i, (x, y) in enumerate(small_positions):
            self.base_stations.append(BaseStation(x, y, 'small', len(self.base_stations), capacity=500))
        
        # Create user equipment
        for i in range(50):
            x = np.random.uniform(0, 20000)
            y = np.random.uniform(0, 20000)
            velocity = np.random.uniform(0, 20)  # m/s winning
            direction = np.random.uniform(0, 2*np.pi)
            self.ues.append(UserEquipment(i, x, y, velocity, direction))
    
    def prepare_ml_models(self):
        """Generate training data and prepare ML models"""
        print("Generating training data for ML models...")
        
        # Simulate traffic patterns for training
        for i in range(500):
            hour = i % 24
            weekday = i // 24 % 7
            
            # Traffic varies by time of day
            base_traffic = 50
            if 9 <= hour <= 17:  # Business hours
                traffic = base_traffic + np.random.uniform(20, 40)
            elif 18 <= hour <= 23:  # Evening
                traffic = base_traffic + np.random.uniform(10, 30)
            else:  # Night
                traffic = base_traffic + np.random.uniform(0, 15)
            
            self.traffic_data.append({
                'time': i,
                'hour': hour,
                'weekday': weekday,
                'traffic_load': traffic
            })
        
        # Prepare handover data
        for _ in range(500):
            self.handover_data.append({
                'rssi_current': np.random.uniform(-90, -70),
                'rssi_target': np.random.uniform(-90, -70),
                'load_current': np.random.uniform(0, 100),
                'load_target': np.random.uniform(0, 100),
                'distance_current': np.random.uniform(0, 5000),
                'distance_target': np.random.uniform(0, 5000),
                'should_handover': np.random.choice([0, 1], p=[0.7, 0.3])
            })
        
        # Prepare QoS data
        for _ in range(500):
            load = np.random.uniform(0, 100)
            distance = np.random.uniform(0, 5000)
            latency = 10 + load * 0.5 + distance * 0.001
            throughput = 100 - load * 0.5 - distance * 0.01
            
            self.qos_data.append({
                'load': load,
                'distance': distance,
                'rssi': np.random.uniform(-90, -70),
                'latency': max(latency, 5),
                'throughput': max(throughput, 10)
            })
        
        # Train traffic predictor
        df_traffic = pd.DataFrame(self.traffic_data)
        X_traffic = df_traffic[['hour', 'weekday']].values
        y_traffic = df_traffic['traffic_load'].values
        self.traffic_predictor = RandomForestRegressor(n_estimators=50)
        self.traffic_predictor.fit(X_traffic, y_traffic)
        
        # Train handover predictor
        df_handover = pd.DataFrame(self.handover_data)
        X_handover = df_handover.drop('should_handover', axis=1).values
        y_handover = df_handover['should_handover'].values
        self.handover_predictor = RandomForestRegressor(n_estimators=50)
        self.handover_predictor.fit(X_handover, y_handover)
        
        # Train QoS predictor
        df_qos = pd.DataFrame(self.qos_data)
        X_qos = df_qos[['load', 'distance', 'rssi']].values
        y_qos = df_qos[['latency', 'throughput']].values
        self.qos_predictor = RandomForestRegressor(n_estimators=50)
        self.qos_predictor.fit(X_qos, y_qos)
        
        # Train anomaly detector
        X_anomaly = np.array([[bs.current_load] for bs in self.base_stations])
        if len(X_anomaly) > 0:
            self.anomaly_detector = IsolationForest(contamination=0.1)
            self.anomaly_detector.fit(X_anomaly)
        
        print("ML models trained successfully!")
    
    def update_ue_positions(self):
        """Update positions of user equipment"""
        for ue in self.ues:
            # Update position based on velocity and direction
            ue.x += ue.velocity * np.cos(ue.direction) * 10  # * 10 for visibility
            ue.y += ue.velocity * np.sin(ue.direction) * 10
            
            # Bounce off boundaries
            if ue.x < 0 or ue.x > 20000:
                ue.direction = np.pi - ue.direction
                ue.x = max(0, min(ue.x, 20000))
            if ue.y < 0 or ue.y > 20000:
                ue.direction = -ue.direction
                ue.y = max(0, min(ue.y, 20000))
    
    def ai_handover_decision(self, ue):
        """AI-based handover decision"""
        if not ue.connected_bs:
            return None
        
        current_bs = ue.connected_bs
        best_bs = None
        best_score = -1000
        
        for bs in self.base_stations:
            if bs.station_id == current_bs.station_id:
                continue
            
            distance = np.sqrt((bs.x - ue.x)**2 + (bs.y - ue.y)**2)
            if distance > bs.coverage_radius:
                continue
            
            rssi = bs.get_rssi(ue.x, ue.y)
            load = bs.current_load
            
            # Use ML model to predict handover decision
            features = np.array([[
                current_bs.get_rssi(ue.x, ue.y),
                rssi,
                current_bs.current_load,
                load,
                np.sqrt((current_bs.x - ue.x)**2 + (current_bs.y - ue.y)**2),
                distance
            ]])
            
            handover_score = self.handover_predictor.predict(features)[0]
            
            if handover_score > best_score and handover_score > 0.5:
                best_score = handover_score
                best_bs = bs
        
        return best_bs
    
    def update_connections(self):
        """Update UE connections and perform handovers"""
        for ue in self.ues:
            # AI-based handover decision
            target_bs = self.ai_handover_decision(ue)
            
            if target_bs:
                # Perform handover
                ue.handover_count += 1
                ue.connected_bs = target_bs
                target_bs.connected_ues.append(ue)
            elif not ue.connected_bs:
                # Initial connection
                best_bs = None
                best_rssi = -1000
                
                for bs in self.base_stations:
                    distance = np.sqrt((bs.x - ue.x)**2 + (bs.y - ue.y)**2)
                    if distance <= bs.coverage_radius:
                        rssi = bs.get_rssi(ue.x, ue.y)
                        if rssi > best_rssi:
                            best_rssi = rssi
                            best_bs = bs
                
                if best_bs:
                    ue.connected_bs = best_bs
                    best_bs.connected_ues.append(ue)
        
        # Periodically clean connected UEs list
        for bs in self.base_stations:
            bs.connected_ues = [ue for ue in bs.connected_ues if ue.connected_bs == bs]
    
    def predict_traffic_load(self):
        """Predict network traffic load using ML"""
        hour = self.time_step % 24
        weekday = self.time_step // 24 % 7
        
        predicted_load = self.traffic_predictor.predict([[hour, weekday]])[0]
        
        # Update actual load
        total_load = sum(len(bs.connected_ues) * 10 for bs in self.base_stations)
        
        return predicted_load, total_load
    
    def detect_anomalies(self):
        """Detect anomalies in network load"""
        if not self.anomaly_detector:
            return []
        
        anomalies = []
        for bs in self.base_stations:
            features = np.array([[bs.current_load]])
            is_anomaly = self.anomaly_detector.predict(features)[0]
            
            if is_anomaly == -1:  # Anomaly detected
                anomalies.append({
                    'time': self.time_step,
                    'station_id': bs.station_id,
                    'station_type': bs.cell_type,
                    'load': bs.current_load,
                    'position': (bs.x, bs.y)
                })
        
        return anomalies
    
    def predict_qos(self, ue):
        """Predict QoS metrics for a UE"""
        if not ue.connected_bs:
            return None, None
        
        load = ue.connected_bs.current_load
        distance = np.sqrt((ue.connected_bs.x - ue.x)**2 + (ue.connected_bs.y - ue.y)**2)
        rssi = ue.connected_bs.get_rssi(ue.x, ue.y)
        
        features = np.array([[load, distance, rssi]])
        qos_prediction = self.qos_predictor.predict(features)[0]
        
        latency = qos_prediction[0]
        throughput = qos_prediction[1]
        
        return latency, throughput
    
    def update(self, frame):
        """Update simulation state"""
        self.time_step += 1
        
        # Update UE positions
        self.update_ue_positions()
        
        # Update connections and handovers
        self.update_connections()
        
        # Update base station load
        for bs in self.base_stations:
            bs.current_load = len(bs.connected_ues) * 10
        
        # Traffic prediction
        predicted_load, actual_load = self.predict_traffic_load()
        self.prediction_log.append({
            'time': self.time_step,
            'predicted_load': predicted_load,
            'actual_load': actual_load
        })
        
        # Anomaly detection
        anomalies = self.detect_anomalies()
        for anomaly in anomalies:
            self.anomaly_log.append(anomaly)
        
        # Log QoS data
        for ue in self.ues:
            if ue.connected_bs:
                latency, throughput = self.predict_qos(ue)
                ue.latency = latency
                ue.throughput = throughput
                
                self.qos_log.append({
                    'time': self.time_step,
                    'ue_id': ue.ue_id,
                    'bs_id': ue.connected_bs.station_id,
                    'latency': latency,
                    'throughput': throughput,
                    'distance': np.sqrt((ue.connected_bs.x - ue.x)**2 + (ue.connected_bs.y - ue.y)**2)
                })
        
        # Update plot
        self.update_plot()
        
        # Save CSV data every 10 frames
        if self.time_step % 10 == 0:
            self.save_csv_data()
    
    def update_plot(self):
        """Update the visualization"""
        ax.clear()
        
        # Plot base stations
        for bs in self.base_stations:
            color = 'blue' if bs.cell_type == 'macro' else 'red'
            size = 300 if bs.cell_type == 'macro' else 100
            ax.scatter(bs.x, bs.y, c=color, s=size, marker='^', alpha=0.7)
            
            # Draw coverage area
            circle = plt.Circle((bs.x, bs.y), bs.coverage_radius, 
                              fill=False, alpha=0.2, linewidth=1, color=color)
            ax.add_patch(circle)
        
        # Plot user equipment
        for ue in self.ues:
            if ue.connected_bs:
                color = 'green'
                # Draw connection line
                ax.plot([ue.x, ue.connected_bs.x], [ue.y, ue.connected_bs.y], 
                       'g-', alpha=0.2, linewidth=0.5)
            else:
                color = 'gray'
            
            ax.scatter(ue.x, ue.y, c=color, s=20, alpha=0.6)
        
        # Add title with network stats
        total_connected = sum(len(bs.connected_ues) for bs in self.base_stations)
        ax.set_title(f'5G/6G Network Simulation - Time: {self.time_step}s | '
                    f'Connected UEs: {total_connected}/50\n'
                    f'ML Traffic Prediction: {self.prediction_log[-1]["predicted_load"]:.1f} | '
                    f'Anomalies Detected: {len([a for a in self.anomaly_log if a["time"] == self.time_step])}',
                    fontsize=10)
        
        ax.set_xlim(-1000, 21000)
        ax.set_ylim(-1000, 21000)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Macro Cell'),
            Patch(facecolor='red', alpha=0.7, label='Small Cell'),
            Patch(facecolor='green', alpha=0.6, label='Connected UE'),
            Patch(facecolor='gray', alpha=0.6, label='Disconnected UE')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def save_csv_data(self):
        """Save simulation data to CSV files"""
        # Network data
        network_df = pd.DataFrame([{
            'time': self.time_step,
            'station_id': bs.station_id,
            'station_type': bs.cell_type,
            'x': bs.x,
            'y': bs.y,
            'load': bs.current_load,
            'connected_ues': len(bs.connected_ues)
        } for bs in self.base_stations])
        network_df.to_csv('network_data.csv', index=False)
        
        # UE data
        ue_df = pd.DataFrame([{
            'time': self.time_step,
            'ue_id': ue.ue_id,
            'x': ue.x,
            'y': ue.y,
            'velocity': ue.velocity,
            'connected_bs': ue.connected_bs.station_id if ue.connected_bs else -1,
            'handover_count': ue.handover_count
        } for ue in self.ues])
        ue_df.to_csv('ue_data.csv', index=False)
        
        # QoS metrics
        if len(self.qos_log) > 0:
            qos_df = pd.DataFrame(self.qos_log)
            qos_df.to_csv('qos_metrics.csv', index=False)
        
        # Predictions
        if len(self.prediction_log) > 0:
            pred_df = pd.DataFrame(self.prediction_log)
            pred_df.to_csv('prediction_results.csv', index=False)
        
        # Anomalies
        if len(self.anomaly_log) > 0:
            anomaly_df = pd.DataFrame(self.anomaly_log)
            anomaly_df.to_csv('anomaly_log.csv', index=False)
    
    def run_simulation(self):
        """Run the animated simulation"""
        global ax
        fig, ax = plt.subplots(figsize=(14, 10))
        
        ani = animation.FuncAnimation(fig, self.update, interval=100, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        
        # Final save
        self.save_csv_data()
        print("\nSimulation complete! CSV files have been generated.")

if __name__ == '__main__':
    print("Starting 5G/6G Network Simulation with AI/ML Enhancements...")
    simulator = NetworkSimulator()
    simulator.run_simulation()

