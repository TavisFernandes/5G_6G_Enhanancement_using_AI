# 5G/6G Network Simulation - Features Overview

## Core Features

### 1. Network Simulation
- **Heterogeneous Network Architecture**
  - 9 Macro cells for wide coverage (5 km radius)
  - 8 Small cells for high-capacity hotspots (500 m radius)
  - 50 User Equipment (UE) with mobility patterns
  
### 2. AI/ML Enhancements

#### Traffic Load Prediction
- **Model**: Random Forest Regressor
- **Features**: Hour of day, Day of week
- **Output**: Predicted network traffic load
- **Accuracy**: Monitors prediction vs actual load

#### AI-Based Handover Decisions
- **Model**: Random Forest Regressor
- **Features**: 
  - Current and target cell RSSI
  - Load of current and target cells
  - Distance to cells
- **Output**: Intelligent handover decisions
- **Benefit**: Reduces unnecessary handovers, improves QoS

#### Anomaly Detection
- **Model**: Isolation Forest
- **Features**: Base station load patterns
- **Output**: Detects unusual network patterns
- **Use Cases**: Overload detection, equipment failure detection

#### QoS Prediction
- **Model**: Random Forest Regressor (Multi-output)
- **Features**: Load, distance, RSSI
- **Output**: Predicts latency and throughput
- **Metrics**:
  - Latency (ms)
  - Throughput (Mbps)

### 3. Visualization & Animation
- Real-time animated network simulation
- Visual representation of:
  - Base station locations and coverage areas
  - User equipment positions and movement
  - Active connections
  - Network topology

### 4. Data Logging
All simulation data is logged to CSV files:
- Network state and metrics
- UE positions and connections
- QoS measurements
- ML predictions
- Detected anomalies

## Technical Implementation

### Network Models
- **Path Loss**: Simplified Friis equation
- **RSSI Calculation**: Based on transmission power and path loss
- **Mobility**: Random walk model for UE movement
- **Connection**: Best signal strength algorithm with AI handover

### Machine Learning Models
All models use scikit-learn:
- **Training**: Pre-generated synthetic data
- **Features**: Network telemetry data
- **Performance**: Optimized for real-time prediction

### Data Flow
1. UE positions updated based on velocity
2. Signal strength calculated for all BS-UE pairs
3. AI handover decision made
4. Connections updated
5. QoS predicted for all active connections
6. Anomalies detected
7. Data logged to CSV
8. Visualization updated

## Performance Metrics

The simulation tracks:
- Connection quality (RSSI)
- Network load distribution
- Handover frequency
- Latency and throughput
- Prediction accuracy
- Anomaly detection rate

## Scalability

The simulation can handle:
- Variable number of base stations
- Variable number of UEs
- Different cell types and configurations
- Extended time periods
- Custom mobility patterns

## Future Enhancements

Potential additions:
- 6G features (terahertz communications, massive MIMO)
- Reinforcement learning for handover optimization
- Federated learning for distributed models
- Beamforming simulation
- Network slicing for different use cases
- Edge computing integration

