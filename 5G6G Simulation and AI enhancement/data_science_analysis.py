"""
Data Science Analysis for 5G Network Simulation
Analyzes CSV files and demonstrates various ML models for network optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
import sys
warnings.filterwarnings('ignore')

# Force output to be unbuffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def load_data():
    """Load all CSV files"""
    print("="*60)
    print("Data Science Analysis - 5G Network Simulation")
    print("="*60)
    print("\nLoading CSV files...")
    
    try:
        network_df = pd.read_csv('network_data.csv')
        ue_df = pd.read_csv('ue_data.csv')
        qos_df = pd.read_csv('qos_metrics.csv')
        prediction_df = pd.read_csv('prediction_results.csv')
        
        print("+ Network data loaded", flush=True)
        print("+ UE data loaded", flush=True)
        print("+ QoS metrics loaded", flush=True)
        print("+ Prediction results loaded", flush=True)
        
        return network_df, ue_df, qos_df, prediction_df
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run quick_demo.py first to generate CSV files")
        return None, None, None, None

def analyze_qos_data(qos_df):
    """Analyze QoS metrics with various ML models"""
    print("\n" + "="*60)
    print("Analyzing QoS Metrics with ML Models")
    print("="*60)
    
    if qos_df is None or len(qos_df) < 10:
        print("Insufficient data for analysis")
        return
    
    # Prepare features and targets
    features = ['distance']
    X = qos_df[features].values
    y_latency = qos_df['latency'].values
    y_throughput = qos_df['throughput'].values
    
    # Split data
    X_train, X_test, y_latency_train, y_latency_test = train_test_split(
        X, y_latency, test_size=0.2, random_state=42
    )
    
    X_train2, X_test2, y_throughput_train, y_throughput_test = train_test_split(
        X, y_throughput, test_size=0.2, random_state=42
    )
    
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr'),
        'Polynomial Regression': LinearRegression(),  # Will use polynomial features
        'Decision Tree': DecisionTreeRegressor(max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
        'AdaBoost': AdaBoostRegressor(n_estimators=50),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
    }
    
    # Train and evaluate models for latency
    print("\n--- Latency Prediction Models ---")
    latency_results = {}
    
    for name, model in models.items():
        # Handle polynomial regression specially
        if name == 'Polynomial Regression':
            poly_features = PolynomialFeatures(degree=3)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
            model.fit(X_train_poly, y_latency_train)
            y_pred = model.predict(X_test_poly)
        elif name == 'Logistic Regression':
            # Convert to classification problem for logistic regression
            y_train_class = (y_latency_train > np.median(y_latency_train)).astype(int)
            y_test_class = (y_latency_test > np.median(y_latency_train)).astype(int)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train_class)
            y_pred_class = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            # Convert back to regression prediction
            y_pred = y_pred_proba * (y_latency_test.max() - y_latency_test.min()) + y_latency_test.min()
        else:
            # Scale features for neural network
            if name == 'Neural Network':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_latency_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_latency_train)
                y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_latency_test, y_pred)
        r2 = r2_score(y_latency_test, y_pred)
        mae = mean_absolute_error(y_latency_test, y_pred)
        mape = mean_absolute_percentage_error(y_latency_test, y_pred)
        
        latency_results[name] = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"{name:30s}: R²={r2:.3f}, MAE={mae:.2f}ms, MAPE={mape:.2f}%")
    
    # Train and evaluate models for throughput
    print("\n--- Throughput Prediction Models ---")
    throughput_results = {}
    
    for name, model in models.items():
        # Handle polynomial regression specially
        if name == 'Polynomial Regression':
            poly_features = PolynomialFeatures(degree=3)
            X_train_poly = poly_features.fit_transform(X_train2)
            X_test_poly = poly_features.transform(X_test2)
            model.fit(X_train_poly, y_throughput_train)
            y_pred = model.predict(X_test_poly)
        elif name == 'Logistic Regression':
            # Convert to classification problem for logistic regression
            y_train_class = (y_throughput_train > np.median(y_throughput_train)).astype(int)
            y_test_class = (y_throughput_test > np.median(y_throughput_train)).astype(int)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train2)
            X_test_scaled = scaler.transform(X_test2)
            model.fit(X_train_scaled, y_train_class)
            y_pred_class = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            # Convert back to regression prediction
            y_pred = y_pred_proba * (y_throughput_test.max() - y_throughput_test.min()) + y_throughput_test.min()
        else:
            # Scale features for neural network
            if name == 'Neural Network':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train2)
                X_test_scaled = scaler.transform(X_test2)
                model.fit(X_train_scaled, y_throughput_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train2, y_throughput_train)
                y_pred = model.predict(X_test2)
        
        mse = mean_squared_error(y_throughput_test, y_pred)
        r2 = r2_score(y_throughput_test, y_pred)
        mae = mean_absolute_error(y_throughput_test, y_pred)
        mape = mean_absolute_percentage_error(y_throughput_test, y_pred)
        
        throughput_results[name] = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"{name:30s}: R²={r2:.3f}, MAE={mae:.2f}Mbps, MAPE={mape:.2f}%")
    
    # Create visualizations
    create_ml_comparison_plots(latency_results, throughput_results, 
                               y_latency_test, y_throughput_test, X_test)
    
    # Best model selection
    best_latency = min(latency_results.items(), key=lambda x: x[1]['MAE'])
    best_throughput = min(throughput_results.items(), key=lambda x: x[1]['MAE'])
    
    print(f"\n+ Best Latency Model: {best_latency[0]} (MAE: {best_latency[1]['MAE']:.2f}ms)")
    print(f"+ Best Throughput Model: {best_throughput[0]} (MAE: {best_throughput[1]['MAE']:.2f}Mbps)")
    
    # Print performance summary table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<30s} {'R² (Lat)':<12s} {'MAE (Lat)':<12s} {'R² (Thr)':<12s} {'MAE (Thr)':<12s}")
    print("-"*80)
    for name in latency_results.keys():
        lat_r2 = latency_results[name]['R2']
        lat_mae = latency_results[name]['MAE']
        thr_r2 = throughput_results[name]['R2']
        thr_mae = throughput_results[name]['MAE']
        print(f"{name:<30s} {lat_r2:<12.3f} {lat_mae:<12.2f} {thr_r2:<12.3f} {thr_mae:<12.2f}")
    print("="*80)

def create_ml_comparison_plots(latency_results, throughput_results, 
                               y_latency_test, y_throughput_test, X_test):
    """Create comparison plots for different ML models"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Model R² scores comparison for latency
    ax1 = plt.subplot(3, 3, 1)
    model_names = list(latency_results.keys())
    r2_scores = [latency_results[m]['R2'] for m in model_names]
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', 
              '#ff6b6b', '#ffa500', '#9b59b6', '#1abc9c', '#e74c3c']
    bars = ax1.barh(model_names, r2_scores, color=colors[:len(model_names)])
    ax1.set_xlabel('R² Score', fontsize=10)
    ax1.set_title('Latency - R² Scores (Higher is Better)', fontsize=10, fontweight='bold')
    ax1.set_xlim([min(0, min(r2_scores))-0.1, 1])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{r2_scores[i]:.3f}', ha='left', va='center', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Model R² scores comparison for throughput
    ax2 = plt.subplot(3, 3, 2)
    r2_scores_thru = [throughput_results[m]['R2'] for m in model_names]
    bars2 = ax2.barh(model_names, r2_scores_thru, color=colors[:len(model_names)])
    ax2.set_xlabel('R² Score', fontsize=10)
    ax2.set_title('Throughput - R² Scores (Higher is Better)', fontsize=10, fontweight='bold')
    ax2.set_xlim([min(0, min(r2_scores_thru))-0.1, 1])
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{r2_scores_thru[i]:.3f}', ha='left', va='center', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: MAE comparison
    ax3 = plt.subplot(3, 3, 3)
    mae_latency = [latency_results[m]['MAE'] for m in model_names]
    mae_throughput = [throughput_results[m]['MAE'] for m in model_names]
    x_pos = np.arange(len(model_names))
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, mae_latency, width, label='Latency MAE (ms)', color='#667eea')
    bars2 = ax3.bar(x_pos + width/2, mae_throughput, width, label='Throughput MAE (Mbps)', color='#764ba2')
    ax3.set_ylabel('Mean Absolute Error', fontsize=10)
    ax3.set_title('MAE Comparison (Lower is Better)', fontsize=10, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: MAPE comparison
    ax4 = plt.subplot(3, 3, 4)
    mape_latency = [latency_results[m].get('MAPE', 0) for m in model_names]
    mape_throughput = [throughput_results[m].get('MAPE', 0) for m in model_names]
    bars3 = ax4.bar(x_pos - width/2, mape_latency, width, label='Latency MAPE (%)', color='#f093fb')
    bars4 = ax4.bar(x_pos + width/2, mape_throughput, width, label='Throughput MAPE (%)', color='#4facfe')
    ax4.set_ylabel('Mean Absolute % Error', fontsize=10)
    ax4.set_title('MAPE Comparison (Lower is Better)', fontsize=10, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Model ranking by R² score
    ax5 = plt.subplot(3, 3, 5)
    # Combine R² scores for ranking
    avg_r2_scores = [(name, (latency_results[name]['R2'] + throughput_results[name]['R2'])/2) for name in model_names]
    avg_r2_scores.sort(key=lambda x: x[1], reverse=True)
    names_sorted = [x[0] for x in avg_r2_scores]
    scores_sorted = [x[1] for x in avg_r2_scores]
    bars5 = ax5.barh(names_sorted, scores_sorted, color=colors[:len(model_names)])
    ax5.set_xlabel('Average R² Score', fontsize=10)
    ax5.set_title('Model Ranking by Average R²', fontsize=10, fontweight='bold')
    for i, bar in enumerate(bars5):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2, 
                f'{scores_sorted[i]:.3f}', ha='left', va='center', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Plot 6: Best model predictions vs actual for latency
    ax6 = plt.subplot(3, 3, 6)
    best_model_name = min(latency_results.items(), key=lambda x: x[1]['MAE'])[0]
    best_pred = latency_results[best_model_name]['predictions']
    ax6.scatter(y_latency_test, best_pred, alpha=0.5, color='#667eea')
    ax6.plot([y_latency_test.min(), y_latency_test.max()], 
            [y_latency_test.min(), y_latency_test.max()], 'r--', lw=2)
    ax6.set_xlabel('Actual Latency (ms)', fontsize=10)
    ax6.set_ylabel('Predicted Latency (ms)', fontsize=10)
    ax6.set_title(f'Best Latency Model: {best_model_name}', fontsize=10, fontweight='bold')
    r2 = latency_results[best_model_name]['R2']
    ax6.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax6.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Best model predictions vs actual for throughput
    ax7 = plt.subplot(3, 3, 7)
    best_model_name2 = min(throughput_results.items(), key=lambda x: x[1]['MAE'])[0]
    best_pred2 = throughput_results[best_model_name2]['predictions']
    ax7.scatter(y_throughput_test, best_pred2, alpha=0.5, color='#764ba2')
    ax7.plot([y_throughput_test.min(), y_throughput_test.max()], 
            [y_throughput_test.min(), y_throughput_test.max()], 'r--', lw=2)
    ax7.set_xlabel('Actual Throughput (Mbps)', fontsize=10)
    ax7.set_ylabel('Predicted Throughput (Mbps)', fontsize=10)
    ax7.set_title(f'Best Throughput Model: {best_model_name2}', fontsize=10, fontweight='bold')
    r2_2 = throughput_results[best_model_name2]['R2']
    ax7.text(0.05, 0.95, f'R² = {r2_2:.3f}', transform=ax7.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Feature importance comparison (if available)
    ax8 = plt.subplot(3, 3, 8)
    try:
        rf_model = throughput_results['Random Forest']['model']
        feature_names = ['Distance']
        importances = rf_model.feature_importances_
        bars = ax8.barh(feature_names, importances, color='#43e97b')
        ax8.set_xlabel('Importance', fontsize=10)
        ax8.set_title('Random Forest - Feature Importance', fontsize=10, fontweight='bold')
        for bar in bars:
            width = bar.get_width()
            ax8.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        ax8.grid(True, alpha=0.3, axis='x')
    except:
        ax8.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Feature Importance', fontsize=10, fontweight='bold')
    
    # Plot 9: Residual plot for best model
    ax9 = plt.subplot(3, 3, 9)
    residuals = y_latency_test - best_pred
    ax9.scatter(best_pred, residuals, alpha=0.5, color='#ff6b6b')
    ax9.axhline(y=0, color='red', linestyle='--', lw=2)
    ax9.set_xlabel('Predicted Latency (ms)', fontsize=10)
    ax9.set_ylabel('Residuals', fontsize=10)
    ax9.set_title(f'Residual Plot: {best_model_name}', fontsize=10, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_models_comparison.png', dpi=150, bbox_inches='tight')
    print("\n+ Saved: ml_models_comparison.png")
    print("+ To view graphs: Open 'view_graphs.html' in your browser")
    print("+ Or run: python open_graphs.py")
    plt.close()

def analyze_traffic_prediction(prediction_df):
    """Analyze traffic prediction accuracy"""
    print("\n" + "="*60)
    print("Analyzing Traffic Prediction Models")
    print("="*60)
    
    if prediction_df is None or len(prediction_df) < 10:
        print("Insufficient data for analysis")
        return
    
    # Calculate prediction error
    prediction_df['error'] = abs(prediction_df['predicted_load'] - prediction_df['actual_load'])
    prediction_df['error_percent'] = (prediction_df['error'] / prediction_df['actual_load']) * 100
    
    # Overall statistics
    print(f"\nOverall Prediction Statistics:")
    print(f"  Mean Error: {prediction_df['error'].mean():.2f}")
    print(f"  Median Error: {prediction_df['error'].median():.2f}")
    print(f"  Accuracy (within 10%): {(prediction_df['error_percent'] < 10).sum() / len(prediction_df) * 100:.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predicted vs Actual
    axes[0, 0].plot(prediction_df['time'], prediction_df['predicted_load'], 
                   label='Predicted', color='#667eea', linewidth=2)
    axes[0, 0].plot(prediction_df['time'], prediction_df['actual_load'], 
                   label='Actual', color='#764ba2', linewidth=2, alpha=0.7)
    axes[0, 0].fill_between(prediction_df['time'], prediction_df['predicted_load'], 
                           prediction_df['actual_load'], alpha=0.2)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Traffic Load')
    axes[0, 0].set_title('Traffic Load: Predicted vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction Error Over Time
    axes[0, 1].plot(prediction_df['time'], prediction_df['error'], 
                   color='#f093fb', linewidth=2)
    axes[0, 1].axhline(y=prediction_df['error'].mean(), color='red', 
                      linestyle='--', label=f'Mean: {prediction_df["error"].mean():.2f}')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Prediction Error')
    axes[0, 1].set_title('Prediction Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(prediction_df['error'], bins=30, color='#4facfe', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(prediction_df['error'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {prediction_df["error"].mean():.2f}')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Accuracy by Time
    prediction_df['accurate'] = (prediction_df['error_percent'] < 10).astype(int)
    window_size = 5
    rolling_accuracy = prediction_df['accurate'].rolling(window=window_size).mean() * 100
    axes[1, 1].plot(prediction_df['time'], rolling_accuracy, 
                   color='#43e97b', linewidth=2)
    axes[1, 1].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% Target')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Rolling Accuracy (%)')
    axes[1, 1].set_title(f'Rolling Accuracy (Window: {window_size})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('traffic_prediction_analysis.png', dpi=150, bbox_inches='tight')
    print("+ Saved: traffic_prediction_analysis.png")
    plt.close()

def analyze_network_topology(network_df):
    """Analyze network topology and load distribution"""
    print("\n" + "="*60)
    print("Analyzing Network Topology")
    print("="*60)
    
    if network_df is None or len(network_df) == 0:
        print("Insufficient data for analysis")
        return
    
    # Statistics
    print(f"\nNetwork Statistics:")
    print(f"  Total Base Stations: {network_df['station_id'].nunique()}")
    print(f"  Macro Cells: {(network_df['station_type'] == 'macro').sum()}")
    print(f"  Small Cells: {(network_df['station_type'] == 'small').sum()}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Load distribution by cell type
    macro_loads = network_df[network_df['station_type'] == 'macro']['load']
    small_loads = network_df[network_df['station_type'] == 'small']['load']
    axes[0, 0].hist(macro_loads, bins=20, alpha=0.7, label='Macro', color='#2196F3', edgecolor='black')
    axes[0, 0].hist(small_loads, bins=20, alpha=0.7, label='Small', color='#F44336', edgecolor='black')
    axes[0, 0].set_xlabel('Load')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Load Distribution by Cell Type')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Network topology
    macro_stations = network_df[network_df['station_type'] == 'macro']
    small_stations = network_df[network_df['station_type'] == 'small']
    axes[0, 1].scatter(macro_stations['x'], macro_stations['y'], c='blue', 
                      s=300, marker='^', label='Macro', alpha=0.7, edgecolors='black')
    axes[0, 1].scatter(small_stations['x'], small_stations['y'], c='red', 
                      s=100, marker='^', label='Small', alpha=0.7, edgecolors='black')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].set_title('Network Topology')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # Plot 3: Connected UEs distribution
    axes[1, 0].hist(network_df['connected_ues'], bins=20, color='#4CAF50', 
                   edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(network_df['connected_ues'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {network_df["connected_ues"].mean():.1f}')
    axes[1, 0].set_xlabel('Connected UEs')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Connected UEs Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Load heatmap
    scatter = axes[1, 1].scatter(network_df['x'], network_df['y'], 
                                c=network_df['load'], s=100, cmap='RdYlGn', 
                                alpha=0.7, edgecolors='black')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    axes[1, 1].set_title('Network Load Heatmap')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[1, 1], label='Load')
    
    plt.tight_layout()
    plt.savefig('network_topology_analysis.png', dpi=150, bbox_inches='tight')
    print("+ Saved: network_topology_analysis.png")
    plt.close()

def main():
    """Main analysis function"""
    # Load data
    network_df, ue_df, qos_df, prediction_df = load_data()
    
    if network_df is None:
        return
    
    # Run analyses
    analyze_network_topology(network_df)
    analyze_qos_data(qos_df)
    analyze_traffic_prediction(prediction_df)
    
    print("\n" + "="*60)
    print("Data Science Analysis Complete!")
    print("="*60)
    print("\nGenerated Files:")
    print("  - ml_models_comparison.png (SAS-style model comparison with 9 plots)")
    print("  - traffic_prediction_analysis.png")
    print("  - network_topology_analysis.png")
    print("\nModels Analyzed:")
    print("  1. Linear Regression")
    print("  2. Ridge Regression")
    print("  3. Lasso Regression")
    print("  4. Elastic Net")
    print("  5. Logistic Regression")
    print("  6. Decision Tree")
    print("  7. Random Forest")
    print("  8. Gradient Boosting")
    print("  9. AdaBoost")
    print("  10. Neural Network (MLP)")
    print("\nMetrics Used:")
    print("  - R² Score (Coefficient of Determination)")
    print("  - MAE (Mean Absolute Error)")
    print("  - MAPE (Mean Absolute Percentage Error)")
    print("\nNext Steps:")
    print("  1. Review the generated plots")
    print("  2. Consider using Random Forest or Gradient Boosting for QoS prediction")
    print("  3. Optimize hyperparameters for better performance")
    print("  4. Integrate best models into network simulation")

if __name__ == '__main__':
    main()

