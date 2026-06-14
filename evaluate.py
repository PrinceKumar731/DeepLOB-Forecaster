import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up matplotlib style for a professional dark look matching the theme
plt.style.use('dark_background')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#2f363f'
plt.rcParams['grid.color'] = '#1f242b'

# Ensure root folder is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.features.feature_extractor import extract_basic_features, add_rolling_features, generate_labels
from src.models.baseline_lgb import LightGBMModel
from src.backtest.simulator import LOBBacktester

# Helper to plot a nice confusion matrix using matplotlib
def plot_confusion_matrix(cm, classes, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    print("Starting LOB Model Evaluation Pipeline...")
    
    # 1. Load recorded LOB dataset
    data_path = "data/raw/btcusdt_lob_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Recorded data file not found at {data_path}")
        sys.exit(1)
        
    print(f"Loading LOB dataset from {data_path}...")
    df_lob = pd.read_csv(data_path)
    df_lob['mid_price'] = (df_lob['bid_p_0'] + df_lob['ask_p_0']) / 2.0
    N = len(df_lob)
    print(f"Loaded {N} limit order book ticks.")

    # 2. Extract features and labels
    print("Extracting order book microstructural features...")
    vector_cols = []
    for i in range(10):
        vector_cols.extend([f"bid_p_{i}", f"bid_q_{i}", f"ask_p_{i}", f"ask_q_{i}"])
    lob_states = df_lob[vector_cols].values
    
    df_feats = extract_basic_features(lob_states, k=10)
    df_feats['timestamp'] = df_lob['timestamp'].values
    df_feats['bid_p_0'] = df_lob['bid_p_0'].values
    df_feats['ask_p_0'] = df_lob['ask_p_0'].values
    df_feats = add_rolling_features(df_feats)
    df_feats['label'] = generate_labels(df_feats['mid_price'].values, horizon=10, threshold=0.0001)

    # Filter out invalid labels (-1) at the boundary
    valid_mask = df_feats['label'] != -1
    y_true = df_feats['label'].values[valid_mask]
    
    # 3. Evaluate LightGBM Baseline Model
    lgb_path = "checkpoints/lgb_model.joblib"
    lgb_preds = None
    if os.path.exists(lgb_path):
        print("Loading LightGBM model...")
        lgb_model = LightGBMModel()
        lgb_model.load(lgb_path)
        
        # Prepare tabular data
        X_eval, _ = lgb_model.prepare_data(df_feats)
        lgb_preds = lgb_model.predict(X_eval)
        print("LightGBM predictions computed.")
    else:
        print("Warning: LightGBM model checkpoint not found.")

    # 4. Evaluate DeepLOB CNN-LSTM Model
    deeplob_path = "checkpoints/deeplob_best.pth"
    scaler_path = "checkpoints/scaler.npz"
    deeplob_preds = None
    
    if os.path.exists(deeplob_path) and os.path.exists(scaler_path):
        print("Loading DeepLOB PyTorch model...")
        try:
            import torch
            from torch.utils.data import DataLoader
            from src.models.deeplob import DeepLOB
            from src.models.trainer import DeepLOBTrainer, LOBDataset
            
            # Load PyTorch model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = DeepLOB(lookback_window=100, num_features=40, num_classes=3)
            trainer = DeepLOBTrainer(model=model, device=device, checkpoint_dir="checkpoints")
            trainer.load_model("checkpoints")
            
            # Scale raw sequence features
            x_raw = df_lob[vector_cols].values
            x_scaled = trainer.scale_data(x_raw)
            y_labels = df_feats['label'].values
            
            # Build sequence dataset loader
            dataset = LOBDataset(x_scaled, y_labels, lookback=100)
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            
            _, test_acc, preds, targets = trainer.evaluate(loader)
            
            # We pad the beginning since DeepLOB requires a lookback window of 100 ticks
            # The predictions match targets at index-mapped valid positions
            # Let's align deepLOB predictions to the length of df_feats
            deeplob_preds_aligned = np.ones(len(df_feats)) # default flat (1)
            
            # Retrieve lookback offset and valid indices mapping
            for idx, start_idx in enumerate(dataset.valid_indices):
                target_idx = start_idx + dataset.lookback - 1
                deeplob_preds_aligned[target_idx] = preds[idx]
                
            deeplob_preds = deeplob_preds_aligned[valid_mask]
            print(f"DeepLOB predictions computed. Device: {device.upper()}")
        except Exception as e:
            print(f"Error evaluating DeepLOB PyTorch model: {e}")
    else:
        print("Warning: DeepLOB PyTorch weights or scaler not found.")

    # Create reports folder
    os.makedirs("reports", exist_ok=True)
    
    # 5. Compile and Print Classification Metrics
    results_summary = []
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    if lgb_preds is not None:
        acc = accuracy_score(y_true, lgb_preds)
        report = classification_report(y_true, lgb_preds, target_names=['Down', 'Flat', 'Up'], output_dict=True)
        results_summary.append({
            "Model": "LightGBM",
            "Accuracy": acc,
            "F1-Down": report['Down']['f1-score'],
            "F1-Flat": report['Flat']['f1-score'],
            "F1-Up": report['Up']['f1-score']
        })
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, lgb_preds)
        plot_confusion_matrix(cm, ['Down', 'Flat', 'Up'], "LightGBM Confusion Matrix", "reports/lgb_confusion_matrix.png")
        
    if deeplob_preds is not None:
        acc = accuracy_score(y_true, deeplob_preds)
        report = classification_report(y_true, deeplob_preds, target_names=['Down', 'Flat', 'Up'], output_dict=True)
        results_summary.append({
            "Model": "DeepLOB (PyTorch)",
            "Accuracy": acc,
            "F1-Down": report['Down']['f1-score'],
            "F1-Flat": report['Flat']['f1-score'],
            "F1-Up": report['Up']['f1-score']
        })
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, deeplob_preds)
        plot_confusion_matrix(cm, ['Down', 'Flat', 'Up'], "DeepLOB Confusion Matrix", "reports/deeplob_confusion_matrix.png")

    # Display Metrics Table
    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        print("\n=== Model Classification Performance ===")
        print(df_summary.to_string(index=False))
        df_summary.to_markdown("reports/evaluation_summary.md", index=False)
        print("\nClassification reports saved to reports/evaluation_summary.md")
    else:
        print("No models evaluated. Check checkpoints.")
        sys.exit(1)

    # 6. Generate Backtesting Returns Plot
    print("\nRunning simulated backtests to compare trading returns...")
    backtester = LOBBacktester(
        initial_capital=100000.0,
        latency_ms=50,
        transaction_fee=0.0005,
        slippage=0.0001,
        tick_interval_ms=100
    )
    
    plt.figure(figsize=(10, 6))
    
    # Baseline Buy & Hold (Buy at start, hold to end)
    mid_prices = df_feats['mid_price'].values
    bh_returns = (mid_prices / mid_prices[0] - 1) * 100
    plt.plot(df_feats['timestamp'], bh_returns, label="Buy & Hold BTC", color='#848e9c', alpha=0.5, linestyle='--')

    # LightGBM Equity
    if lgb_preds is not None:
        lgb_preds_full = np.ones(len(df_feats))
        lgb_preds_full[valid_mask] = lgb_preds
        metrics, equity_curve, _ = backtester.run(df_feats, lgb_preds_full)
        lgb_returns = (equity_curve / 100000.0 - 1) * 100
        plt.plot(df_feats['timestamp'], lgb_returns, label=f"LightGBM Strategy ({metrics['total_return_pct']:.2f}% Return)", color='#f0b90b', linewidth=2)
        print(f"LightGBM Backtest - Return: {metrics['total_return_pct']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}")

    # DeepLOB Equity
    if deeplob_preds is not None:
        deeplob_preds_full = np.ones(len(df_feats))
        deeplob_preds_full[valid_mask] = deeplob_preds
        metrics, equity_curve, _ = backtester.run(df_feats, deeplob_preds_full)
        deeplob_returns = (equity_curve / 100000.0 - 1) * 100
        plt.plot(df_feats['timestamp'], deeplob_returns, label=f"DeepLOB Strategy ({metrics['total_return_pct']:.2f}% Return)", color='#0ecb81', linewidth=2)
        print(f"DeepLOB Backtest - Return: {metrics['total_return_pct']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}")

    plt.title("LOB Trading Strategy Returns Comparison")
    plt.xlabel("Timestamp (Unix Seconds)")
    plt.ylabel("Cumulative Returns (%)")
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/backtest_returns.png", dpi=300)
    plt.close()
    print("Backtest returns curve saved to reports/backtest_returns.png")

    # 7. Generate Prediction Timeline Chart
    print("Plotting prediction signals overlay...")
    plt.figure(figsize=(12, 6))
    
    # We plot the last 300 ticks to zoom in and see the trends clearly
    zoom_len = min(300, len(df_feats) - 100)
    zoom_slice = slice(-zoom_len, None)
    
    zoom_times = df_feats['timestamp'].iloc[zoom_slice].values
    zoom_prices = df_feats['mid_price'].iloc[zoom_slice].values
    
    plt.plot(zoom_times, zoom_prices, color='#848e9c', label="BTC Mid Price ($)", linewidth=1.5)
    
    # Overlays for signals
    if lgb_preds is not None:
        lgb_preds_zoom = lgb_preds_full[zoom_slice]
        
        # Up signals
        up_idx = np.where(lgb_preds_zoom == 2)[0]
        if len(up_idx) > 0:
            plt.scatter(zoom_times[up_idx], zoom_prices[up_idx], color='#0ecb81', marker='^', s=40, label="LGB Buy Signal", zorder=3)
            
        # Down signals
        down_idx = np.where(lgb_preds_zoom == 0)[0]
        if len(down_idx) > 0:
            plt.scatter(zoom_times[down_idx], zoom_prices[down_idx], color='#f6465d', marker='v', s=40, label="LGB Sell Signal", zorder=3)

    plt.title("LOB Price Movements & Live Signal Triggers (Zoomed View)")
    plt.xlabel("Timestamp")
    plt.ylabel("Price ($)")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/prediction_trends.png", dpi=300)
    plt.close()
    print("Prediction overlay timeline saved to reports/prediction_trends.png")
    print("\nEvaluation successfully completed. All graphs generated in reports/ directory.")

if __name__ == "__main__":
    main()
