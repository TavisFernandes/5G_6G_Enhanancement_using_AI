# 📊 How to View the Data Science Analysis Graphs

## Quick Start

After running `python data_science_analysis.py`, view the graphs by:

### Option 1: Automatic (Recommended)
```bash
python open_graphs.py
```
This will automatically open the graph viewer in your default web browser.

### Option 2: Manual
Simply open `view_graphs.html` in any web browser:
- Double-click the file
- Or drag and drop it into your browser

---

## 📈 What You'll See

The graph viewer displays **4 comprehensive analysis visualizations**:

### 1. ML Models Comparison (SAS-Style)
- **9 different visualizations** comparing 10 machine learning models:
  - R² Scores for Latency and Throughput
  - MAE (Mean Absolute Error) Comparison
  - MAPE (Mean Absolute Percentage Error) Comparison
  - Model Ranking by Average R²
  - Best Model Predictions vs Actual
  - Feature Importance
  - Residual Plots

### 2. Traffic Prediction Analysis
- Predicted vs Actual traffic load
- Prediction error over time
- Error distribution histogram
- Rolling accuracy metrics

### 3. Network Topology Analysis
- Load distribution by cell type (Macro vs Small cells)
- Network topology map
- Connected UEs distribution
- Network load heatmap

### 4. QoS Analysis
- Quality of Service metrics
- Latency and throughput analysis

---

## 🤖 Models Included

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Polynomial Regression (degree=3)
6. Decision Tree
7. Random Forest
8. Gradient Boosting
9. AdaBoost
10. Neural Network (MLP)

---

## 📊 Metrics Explained

- **R² Score**: Higher is better (ideal = 1.0)
  - Measures how well the model fits the data
  
- **MAE**: Lower is better
  - Mean Absolute Error (average prediction error)
  
- **MAPE**: Lower is better
  - Mean Absolute Percentage Error (% error)

---

## 🎯 Current Best Models

- **Best Latency Model**: Gradient Boosting (MAE: 11.33ms)
- **Best Throughput Model**: Gradient Boosting (MAE: 10.83Mbps)
- **Traffic Prediction Accuracy**: 100%

---

## 💡 Tips

- **Hover** over graphs for better visibility
- **Right-click** any graph to save it as an image
- **Zoom in** by hovering over the graph (it will scale slightly)
- Graphs are stored as PNG files in the project directory

---

## 🔧 Troubleshooting

### Graphs not showing?

1. Make sure you've run:
   ```bash
   python quick_demo.py
   python data_science_analysis.py
   ```

2. Check that these files exist:
   - `ml_models_comparison.png`
   - `traffic_prediction_analysis.png`
   - `network_topology_analysis.png`
   - `qos_analysis.png`

3. If files are missing, regenerate them:
   ```bash
   python data_science_analysis.py
   ```

### Browser not opening?

- Manually open `view_graphs.html` in your browser
- The HTML file uses local image paths, so it should work offline

---

## ✨ Features

- Beautiful, modern web interface
- Interactive hover effects
- Responsive design (works on desktop and mobile)
- All graphs in one place for easy comparison
- Professional SAS-style visualizations


