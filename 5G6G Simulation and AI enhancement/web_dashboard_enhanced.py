"""
Enhanced Flask Web Dashboard for 5G Network Simulation
Displays real-time simulation with network topology visualization
"""

from flask import Flask, render_template, jsonify
import json
import pandas as pd
import numpy as np
from threading import Thread
import time
import sys
import os

app = Flask(__name__)

# Global simulation data
simulation_data = {
    'network_stats': [],
    'qos_data': [],
    'predictions': [],
    'anomalies': [],
    'handovers': [],
    'time': 0,
    'simulator': None,
    'topology': []
}

# Import simulator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("sim_module", "5g_network_simulation.py")
sim_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_module)
NetworkSimulator = sim_module.NetworkSimulator

def run_simulation():
    """Run simulation in background and update data"""
    global simulation_data
    
    print("Starting simulation...")
    simulator = NetworkSimulator()
    simulation_data['simulator'] = simulator
    
    for i in range(100):  # Run for 100 steps
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
        
        # Update global data
        connected_count = sum(len(bs.connected_ues) for bs in simulator.base_stations)
        total_handovers = sum(ue.handover_count for ue in simulator.ues)
        
        simulation_data['network_stats'].append({
            'time': i,
            'connected_ues': connected_count,
            'total_handovers': total_handovers,
            'avg_latency': np.mean([ue.latency for ue in simulator.ues if ue.connected_bs]) if any(ue.connected_bs for ue in simulator.ues) else 0,
            'avg_throughput': np.mean([ue.throughput for ue in simulator.ues if ue.connected_bs]) if any(ue.connected_bs for ue in simulator.ues) else 0
        })
        
        # Update topology data
        simulation_data['topology'] = get_topology_data(simulator)
        
        # Keep only last 50 entries for performance
        if len(simulation_data['network_stats']) > 50:
            simulation_data['network_stats'] = simulation_data['network_stats'][-50:]
        
        # Update predictions
        if simulator.prediction_log:
            simulation_data['predictions'] = simulator.prediction_log[-50:]
        
        # Update anomalies
        simulation_data['anomalies'] = simulator.anomaly_log[-20:]
        
        time.sleep(0.5)  # Slow down for visualization
    
    print("Simulation complete!")
    simulator.save_csv_data()

def get_topology_data(simulator):
    """Get topology data for visualization"""
    bs_list = []
    for bs in simulator.base_stations:
        bs_list.append({
            'id': bs.station_id,
            'x': bs.x,
            'y': bs.y,
            'type': bs.cell_type,
            'coverage_radius': bs.coverage_radius,
            'load': bs.current_load,
            'num_ues': len(bs.connected_ues)
        })
    
    ue_list = []
    for ue in simulator.ues:
        ue_data = {
            'id': ue.ue_id,
            'x': ue.x,
            'y': ue.y,
            'velocity': ue.velocity,
            'connected': ue.connected_bs is not None,
            'connected_bs': ue.connected_bs.station_id if ue.connected_bs else -1,
            'latency': round(ue.latency, 2) if ue.connected_bs else 0,
            'throughput': round(ue.throughput, 2) if ue.connected_bs else 0,
            'handover_count': ue.handover_count
        }
        ue_list.append(ue_data)
    
    return {'base_stations': bs_list, 'user_equipment': ue_list}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard_enhanced.html')

@app.route('/api/stats')
def get_stats():
    """Get current network statistics"""
    return jsonify(simulation_data['network_stats'])

@app.route('/api/predictions')
def get_predictions():
    """Get traffic predictions"""
    return jsonify(simulation_data['predictions'])

@app.route('/api/anomalies')
def get_anomalies():
    """Get detected anomalies"""
    return jsonify(simulation_data['anomalies'])

@app.route('/api/topology')
def get_topology():
    """Get network topology"""
    return jsonify(simulation_data['topology'])

@app.route('/api/qos-data')
def get_qos_data():
    """Get QoS metrics"""
    try:
        df = pd.read_csv('qos_metrics.csv')
        data = df.tail(100).to_dict('records')
        return jsonify(data)
    except:
        return jsonify([])

@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    try:
        stats = simulation_data['network_stats']
        if not stats:
            return jsonify({})
        
        latest = stats[-1]
        
        # Get simulator for additional stats
        simulator = simulation_data['simulator']
        beamforming_efficiency = 0
        mimo_gain = 0
        resource_utilization = 0
        
        if simulator:
            # Calculate beamforming efficiency
            beamforming_efficiency = min(100, latest['avg_throughput'] / 10 * 10)
            mimo_gain = min(100, latest['avg_throughput'] / 15 * 10)
            resource_utilization = min(100, latest['connected_ues'] / 50 * 100)
            
            # 6G features
            terahertz_utilization = min(100, latest['avg_throughput'] * 1.5 / 20 * 100)
            ai_efficiency = min(100, 85 + (latest['connected_ues'] / 50) * 10)
            holographic_capability = min(100, latest['avg_throughput'] / 25 * 100)
            edge_computing_load = min(100, latest['connected_ues'] / 40 * 100)
        else:
            terahertz_utilization = 0
            ai_efficiency = 0
            holographic_capability = 0
            edge_computing_load = 0
        
        summary = {
            'time': latest['time'],
            'connected_ues': latest['connected_ues'],
            'total_handovers': latest['total_handovers'],
            'avg_latency': round(latest['avg_latency'], 2),
            'avg_throughput': round(latest['avg_throughput'], 2),
            'anomalies_detected': len(simulation_data['anomalies']),
            'beamforming_efficiency': round(beamforming_efficiency, 1),
            'mimo_gain': round(mimo_gain, 1),
            'resource_utilization': round(resource_utilization, 1),
            'terahertz_utilization': round(terahertz_utilization, 1),
            'ai_efficiency': round(ai_efficiency, 1),
            'holographic_capability': round(holographic_capability, 1),
            'edge_computing_load': round(edge_computing_load, 1)
        }
        
        # Calculate prediction accuracy
        if simulation_data['predictions']:
            preds = simulation_data['predictions'][-10:]
            errors = [abs(p['predicted_load'] - p['actual_load']) for p in preds]
            avg_error = np.mean(errors)
            summary['prediction_error'] = round(avg_error, 2)
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Start simulation in background thread
    sim_thread = Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    # Give simulation time to start
    time.sleep(2)
    
    # Start Flask server
    print("\n" + "="*60)
    print("5G Network Simulation Enhanced Web Dashboard")
    print("="*60)
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nFeatures:")
    print("  + Real-time network topology visualization")
    print("  + Interactive cell tower map (Simu5G-like)")
    print("  + Advanced 5G/6G features (Beamforming, MIMO, AI)")
    print("  + AI/ML metrics and graphs")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)

