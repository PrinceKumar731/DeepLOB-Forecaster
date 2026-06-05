# High-Frequency Limit Order Book (LOB) Dynamics

This repository provides a framework for analyzing, predicting, and backtesting short-term price movements using Limit Order Book (LOB) data. It includes data ingestion, feature engineering, predictive modeling, a latency-aware trading simulator, and an interactive dashboard.

## Key Features

1. **Data Ingestion**: A live WebSocket client subscribing to Binance market data (e.g., BTCUSDT order book snapshots at 100ms intervals).
2. **Feature Extraction**: Calculates order book imbalance (OBI), micro-price, bid-ask spread, order flow imbalance (OFI), and rolling volatility metrics.
3. **Machine Learning Models**:
   - **DeepLOB (PyTorch)**: A convolutional LSTM neural network (CNN-LSTM) that extracts spatial and temporal features from the order book.
   - **LightGBM**: A fast tabular baseline model utilizing rolling feature statistics.
4. **Latency-Aware Backtesting**: Evaluates trading performance by introducing simulated execution delays, transaction fees, and slippage.
5. **Interactive Dashboard**: A Streamlit interface to visualize live order book depth, track model predictions, and replay backtests.

## Project Structure

```
quantML/
├── config/
│   └── config.yaml             # Hyperparameters and paths
├── src/
│   ├── data/
│   │   └── binance_client.py   # Live WebSocket client and recorder
│   ├── features/
│   │   ├── book_builder.py     # Local order book manager
│   │   └── feature_extractor.py # Feature engineering module
│   ├── models/
│   │   ├── deeplob.py          # PyTorch DeepLOB implementation
│   │   ├── baseline_lgb.py     # LightGBM baseline module
│   │   └── trainer.py          # Dataset loader and training loops
│   ├── backtest/
│   │   └── simulator.py        # Latency-aware backtester
│   └── dashboard/
│       └── app.py              # Streamlit dashboard layout
├── tests/
│   ├── test_book_builder.py    # Unit tests for order book state
│   └── test_feature_extractor.py # Unit tests for feature calculations
├── run_pipeline.py             # Pipeline orchestrator
├── requirements.txt            # Package dependencies
└── README.md                   # Setup and usage guide
```

## Installation

1. Set up a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

All workflows are orchestrated via the `run_pipeline.py` script.

### 1. Run Unit Tests
Verify order book sorting and feature formulas:
```powershell
python run_pipeline.py --mode test
```

### 2. Record Order Book Data
Stream and record live LOB updates from Binance:
```powershell
python run_pipeline.py --mode record --symbol btcusdt --duration 300
```
Data is saved to `data/raw/btcusdt_lob_data.csv`.

### 3. Train Models
Build features and train prediction models. If no recorded data exists, a mock dataset is generated automatically:
- Train LightGBM baseline:
  ```powershell
  python run_pipeline.py --mode train --model lgb
  ```
- Train DeepLOB:
  ```powershell
  python run_pipeline.py --mode train --model deeplob --epochs 10
  ```
Model files are saved to `checkpoints/`.

### 4. Launch the Dashboard
Visualize real-time order book depth, live predictions, and backtests:
```powershell
python run_pipeline.py --mode dashboard
```
Open `http://localhost:8501` in your browser.

## Backtesting and Latency Model

- **Execution Delay**: Latency is modeled as a fixed step offset based on the feed resolution. Signals generated at tick $t$ execute at tick $t + \text{latency steps}$ to account for network delays.
- **Friction**: Incorporates customizable maker/taker fees and slippage.
- **Causality**: Features are computed strictly using historical windows to avoid look-ahead bias.
