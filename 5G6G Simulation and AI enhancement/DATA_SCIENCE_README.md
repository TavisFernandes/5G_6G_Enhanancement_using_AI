# Data Science Analysis - Complete Guide

## Overview

The data science analysis script performs comprehensive ML model comparison on your 5G network simulation data.

## What It Does

### 1. Loads Simulation Data
- `network_data.csv` - Base station information
- `ue_data.csv` - User equipment data
- `qos_metrics.csv` - Quality of Service measurements
- `prediction_results.csv` - ML predictions

### 2. Tests 5 ML Models
- **Linear Regression** - Simple baseline
- **Ridge Regression** - L2 regularization
- **Lasso Regression** - L1 regularization  
- **Random Forest** - Ensemble decision trees
- **Gradient Boosting** - Advanced ensemble

### 3. Generates 3 Analysis Files

#### ml_models_comparison.png
- Model R² scores for latency and throughput
- MAE comparison across all models
- Best model predictions vs actual
- Feature importance analysis

#### traffic_prediction_analysis.png
- Predicted vs actual traffic load
- Prediction error over time
- Error distribution histogram
- Rolling accuracy tracking

#### network_topology_analysis.png
- Load distribution by cell type
- Network topology visualization
- Connected UEs distribution
- Load heatmap

## How to Run

```bash
# First, generate data
python quick_demo.py

# Then, run analysis
python data_science_analysis.py
```

## Results Explained

### Model Performance
The script shows:
- **R² Score**: How well model fits (1.0 = perfect, 0 = baseline)
- **MAE**: Mean Absolute Error (lower is better)
- **Best Model**: Automatically identified

### Typical Results
- **Best Latency Model**: Usually Gradient Boosting or Random Forest
- **Best Throughput Model**: Usually Gradient Boosting
- **Traffic Accuracy**: Shows prediction quality

## Use Cases

1. **Network Optimization**: Identify best ML models for your data
2. **Research**: Compare different ML approaches
3. **Validation**: Verify prediction accuracy
4. **Integration**: Select models to integrate into simulation

## Next Steps

1. Review the generated PNG files
2. Compare model R² scores
3. Check prediction accuracy
4. Integrate best models into web_dashboard
5. Optimize hyperparameters for better performance

---

**Ready to use! No errors, fully tested.** ✅



