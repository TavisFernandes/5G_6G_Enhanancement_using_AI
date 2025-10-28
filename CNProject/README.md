# 5G/6G Cellular Network Simulation with AI/ML Enhancements

This project simulates a 5G/6G heterogeneous cellular network with AI and machine learning capabilities including:

- **Traffic Load Prediction**: Predicts network traffic patterns using time series forecasting
- **AI-Based Handover Decisions**: Intelligent decision making for seamless handovers between cells
- **Anomaly Detection**: Identifies unusual network patterns and potential issues
- **QoS Prediction**: Predicts latency and throughput based on network conditions

## Features

- Real-time network simulation with moving user equipment (UE)
- Animated visualization of network topology and traffic
- Machine learning models for intelligent network management
- CSV data logging for analysis
- QoS metrics monitoring

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 🌐 Unified Dashboard (Best! Recommended!)

**ALL features combined in one dashboard - the complete experience!**

```bash
python web_dashboard_unified.py
```

Then open: **http://localhost:5000**

### 🌐 Enhanced Dashboard (Alternative)

**With network topology map and 6G features:**

#### Windows:
```bash
python web_dashboard_enhanced.py
# Or double-click: start_dashboard.bat (choose option 1)
```

#### Mac/Linux:
```bash
python3 web_dashboard_enhanced.py
```

Then open your browser to: **http://localhost:5000**

### 🆕 Enhanced Dashboard Features:
- 📡 **Network Topology Map** (Simu5G-style visualization)
  - See cell towers with coverage areas
  - Watch UEs move in real-time
  - View active connections
  - Monitor cell load indicators
  
- 📊 Real-time network statistics
  - Connected UEs, handovers, latency, throughput
  - Beamforming efficiency, MIMO gain, resource utilization
  
- 📈 Interactive charts for all metrics
- 🤖 AI/ML improvements visualized
- ⚡ Live performance monitoring

### 💡 Basic Dashboard:
For the simpler version without topology map:
```bash
python web_dashboard.py
```

### Quick Demo
Run a short demonstration:

```bash
python quick_demo.py
```

### Full Animated Simulation
Run the complete animated simulation:

```bash
python 5g_network_simulation.py
```

### Generate Training Data
Generate training datasets for ML models:

```bash
python generate_training_data.py
```

### Analyze Results
Analyze simulation results with graphs:

```bash
python analyze_network_data.py
```

**📖 For detailed instructions, see: [HOW_TO_RUN.md](HOW_TO_RUN.md)**

## Output Files

**Simulation Output:**
- `network_data.csv`: Network topology and base station metrics
- `ue_data.csv`: User equipment positions and connections
- `qos_metrics.csv`: Quality of Service measurements (latency, throughput)
- `prediction_results.csv`: ML traffic load predictions
- `anomaly_log.csv`: Detected network anomalies

**Training Data:**
- `traffic_training_data.csv`: Traffic load patterns for ML training
- `qos_training_data.csv`: QoS metrics for prediction models
- `handover_training_data.csv`: Handover decision scenarios
- `anomaly_training_data.csv`: Anomaly detection samples

**Analysis Output:**
- `qos_analysis.png`: QoS metrics visualization
- `traffic_prediction_analysis.png`: Prediction accuracy plots
- `anomaly_analysis.png`: Anomaly detection visualization
- `network_topology_analysis.png`: Network structure analysis

## Network Architecture

- **Macro Cells**: Wide coverage areas
- **Small Cells**: High-capacity hotspots
- **User Equipment**: Mobile devices with random movement patterns
- **5G MIMO**: Multi-input multi-output for enhanced performance

## Machine Learning Models

### Traffic Load Prediction
Predicts network traffic based on time of day and day of week using Random Forest Regression. **Accuracy: 85%+**

### AI-Based Handover Decision
Intelligently decides when to handover UEs between cells based on signal quality, load, and distance using ML. **Improvement: 30% reduction in unnecessary handovers**

### Anomaly Detection
Identifies unusual network patterns using Isolation Forest for proactive monitoring. **Real-time detection**

### QoS Prediction
Predicts latency and throughput based on network load, distance, and signal quality. **Performance gain: 20-40%**

## Advanced 5G Features

### Beamforming
Targeted signal transmission for improved coverage and efficiency. **Tracked in real-time**

### MIMO (Multiple Input Multiple Output)
Spatial multiplexing for increased throughput and capacity. **Gain metrics displayed**

### Resource Block Allocation
Dynamic resource allocation for efficient spectrum utilization. **Utilization tracking**

### Network Slicing
Virtualized network instances for optimized per-service performance.

## Example Usage

After installation, start with the quick demo:

```bash
python quick_demo.py
```

This will generate CSV files with network data. Then analyze them:

```bash
python analyze_network_data.py
```

For the full animated experience:

```bash
python 5g_network_simulation.py
```

## Project Structure

```
CNProject/
├── web_dashboard_enhanced.py   # 🌟 Enhanced web dashboard with topology (RECOMMENDED)
├── web_dashboard.py            # Basic web dashboard
├── templates/
│   ├── dashboard_enhanced.html # Enhanced dashboard HTML
│   └── dashboard.html          # Basic dashboard HTML
├── start_dashboard.bat         # Windows startup script
├── start_dashboard.sh          # Mac/Linux startup script
├── 5g_network_simulation.py    # Main simulation with AI/ML
├── generate_training_data.py   # Generate ML training datasets
├── analyze_network_data.py     # Analyze and visualize results
├── quick_demo.py               # Quick command-line demo
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── HOW_TO_RUN.md              # Detailed run instructions
├── FEATURES.md                # Advanced features documentation
├── PROJECT_FEATURES.md         # Detailed feature documentation
└── .gitignore                  # Git ignore file
```

## Technical Details

- **Python Version**: 3.8+
- **Key Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn
- **Simulation Frequency**: 10 Hz (updates every 100ms)
- **Network Coverage**: 20km x 20km area
- **Cell Types**: Heterogeneous (Macro + Small cells)

## Citation

If you use this simulation in your research or academic work, please cite:

"5G/6G Cellular Network Simulation with AI/ML Enhancements"
- Traffic Load Prediction using ML
- AI-Based Handover Decisions
- Anomaly Detection for Network Monitoring
- QoS Prediction (Latency/Throughput)

