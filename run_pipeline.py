import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

# Ensure root folder is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def run_tests():
    """Runs the unit tests using pytest."""
    print("Executing Unit Tests...")
    cmd = [os.path.join(".venv", "Scripts", "python"), "-m", "pytest", "tests/"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

def run_dashboard():
    """Launches the Streamlit app dashboard."""
    print("Launching Streamlit Dashboard...")
    cmd = [os.path.join(".venv", "Scripts", "streamlit"), "run", "src/dashboard/app.py"]
    subprocess.run(cmd)

def run_recording(symbol, duration):
    """Subscribes to live Binance websocket stream and records LOB data."""
    print(f"Connecting to Binance live WebSocket feed for {symbol.upper()}...")
    from src.data.binance_client import BinanceLOBClient
    client = BinanceLOBClient(symbol=symbol)
    try:
        client.start(record=True, duration=duration)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        client.flush_records()

def run_training(model_type, epochs):
    """Orchestrates LOB feature extraction and model training."""
    raw_dir = "data/raw"
    csv_files = []
    if os.path.exists(raw_dir):
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        
    if not csv_files:
        print("No recorded historical data found in `data/raw/`.")
        print("Generating mock dataset for training demonstration...")
        from src.dashboard.app import generate_mock_data
        df_lob = generate_mock_data(n_samples=2000)
        os.makedirs(raw_dir, exist_ok=True)
        df_lob.to_csv(os.path.join(raw_dir, "btcusdt_lob_data.csv"), index=False)
        print("Saved btcusdt_lob_data.csv to data/raw/")
    else:
        df_lob = pd.read_csv(os.path.join(raw_dir, csv_files[0]))
        print(f"Loaded recorded dataset: {csv_files[0]} ({len(df_lob)} rows)")

    # Prepare features
    print("Extracting features and engineering lags/rolling metrics...")
    from src.features.feature_extractor import extract_basic_features, add_rolling_features, generate_labels
    
    vector_cols = []
    for i in range(10):
        vector_cols.extend([f"bid_p_{i}", f"bid_q_{i}", f"ask_p_{i}", f"ask_q_{i}"])
    lob_states = df_lob[vector_cols].values
    
    df_feats = extract_basic_features(lob_states, k=10)
    df_feats['timestamp'] = df_lob['timestamp'].values
    df_feats['last_update_id'] = df_lob['last_update_id'].values
    df_feats['bid_p_0'] = df_lob['bid_p_0'].values
    df_feats['ask_p_0'] = df_lob['ask_p_0'].values
    
    df_feats = add_rolling_features(df_feats)
    df_feats['label'] = generate_labels(df_feats['mid_price'].values, horizon=10, threshold=0.0001)

    # Train/Val/Test split (Chronological split to prevent overlap leak)
    N = len(df_feats)
    train_split = int(N * 0.7)
    val_split = int(N * 0.85)
    
    df_train = df_feats.iloc[:train_split]
    df_val = df_feats.iloc[train_split:val_split]
    df_test = df_feats.iloc[val_split:]

    if model_type == "lgb":
        from src.models.baseline_lgb import LightGBMModel
        model = LightGBMModel()
        
        # Prepare tabular data
        X_train, y_train = model.prepare_data(df_train)
        X_val, y_val = model.prepare_data(df_val)
        X_test, y_test = model.prepare_data(df_test)
        
        # Train
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        model.evaluate(X_test, y_test)
        
        # Save
        model.save("checkpoints/lgb_model.joblib")
        
    elif model_type == "deeplob":
        # Check if PyTorch is installed
        try:
            import torch
            from src.models.deeplob import DeepLOB
            from src.models.trainer import DeepLOBTrainer
        except ImportError:
            print("PyTorch is not fully installed yet. Please wait for the requirements to finish installing.")
            sys.exit(1)

        # Prepare arrays
        # Extract features for DeepLOB (which uses the raw 40 sequence features)
        x_train_raw = df_lob.iloc[:train_split][vector_cols].values
        y_train = df_train['label'].values
        
        x_val_raw = df_lob.iloc[train_split:val_split][vector_cols].values
        y_val = df_val['label'].values
        
        x_test_raw = df_lob.iloc[val_split:][vector_cols].values
        y_test = df_test['label'].values
        
        # Instantiate DeepLOB model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Instantiating DeepLOB CNN-LSTM model on device: {device}...")
        
        deeplob = DeepLOB(lookback_window=100, num_features=40, num_classes=3)
        trainer = DeepLOBTrainer(model=deeplob, lr=0.001, device=device, checkpoint_dir="checkpoints")
        
        # Train
        trainer.train(
            x_train_raw, y_train,
            x_val_raw, y_val,
            batch_size=64,
            epochs=epochs,
            early_stopping_rounds=5
        )
        
        # Test evaluation
        from torch.utils.data import DataLoader
        from src.models.trainer import LOBDataset
        
        x_test = trainer.scale_data(x_test_raw)
        test_dataset = LOBDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        val_loss, val_acc, preds, targets = trainer.evaluate(test_loader)
        print(f"\n--- DeepLOB Test Set Performance ---")
        print(f"Test Accuracy: {val_acc:.4f}")
        print(f"Test Loss:     {val_loss:.4f}")
        
        from sklearn.metrics import classification_report
        print(classification_report(targets, preds, target_names=['Down', 'Flat', 'Up']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limit Order Book High-Frequency Dynamics Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["record", "train", "dashboard", "test"],
                        help="Action to perform: record data, train models, launch dashboard, or run tests.")
    parser.add_argument("--symbol", type=str, default="btcusdt", help="Binance trading symbol (default: btcusdt)")
    parser.add_argument("--duration", type=int, default=300, help="Recording duration in seconds (default: 300)")
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "deeplob"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for DeepLOB")

    args = parser.parse_args()

    if args.mode == "test":
        run_tests()
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "record":
        run_recording(args.symbol, args.duration)
    elif args.mode == "train":
        run_training(args.model, args.epochs)
