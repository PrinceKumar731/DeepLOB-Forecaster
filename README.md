# High-Frequency Limit Order Book (LOB) Dynamics

This repository contains an end-to-end framework for analyzing, predicting, and backtesting short-term price movements using structural imbalances in a live Limit Order Book (LOB). It streams real-time tick data from Binance, maintains local book states, extracts microstructural features, trains deep learning (DeepLOB CNN-LSTM) and LightGBM models, runs latency-aware simulated backtests, and displays them on a live Streamlit dashboard.

---

## 🌟 Features

1. **Live LOB Ingestion**: Subscribes to the Binance WebSocket public feeds (e.g. `BTCUSDT` depth snapshots at 100ms intervals) to maintain a synchronized local order book up to 10 levels.
2. **Microstructural Feature Extraction**:
   - **Order Book Imbalance (OBI)**: Volume imbalance between bid and ask sides across different depth levels.
   - **Micro-Price**: Depth-weighted mid-price proxy capturing immediate order book pressure.
   - **Bid-Ask Spread**: Absolute and relative spreads.
   - **Order Flow Imbalance (OFI)**: Tick-by-tick changes in bid and ask depth to measure net order flow velocity.
   - **Rolling Dynamics**: Volatility, average spreads, and cumulative flow velocity over rolling historical windows.
3. **Advanced Machine Learning Models**:
   - **DeepLOB (PyTorch)**: A 2D Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) layer, custom-designed to extract spatial LOB patterns (price-volume levels) and temporal sequences.
   - **LightGBM Baseline**: A fast, tree-based baseline classifier utilizing rolling tabular features.
4. **Latency-Aware Backtester**: Simulates limit/market order execution by applying a configurable network/execution latency (translating to a timestamp index lag) alongside transaction fees and price slippage.
5. **Interactive Analytics Dashboard**: A Streamlit application displaying real-time order book depth charts, live-calculated features, model signal streams, and historical backtesting performance.

---

## 📁 Repository Structure

```
quantML/
├── config/
│   └── config.yaml             # System & model hyperparameters
├── src/
│   ├── data/
│   │   └── binance_client.py   # Live WebSocket client & CSV recorder
│   ├── features/
│   │   ├── book_builder.py     # Local order book state manager
│   │   └── feature_extractor.py # OBI, micro-price, OFI, and rolling features
│   ├── models/
│   │   ├── deeplob.py          # PyTorch DeepLOB (CNN-LSTM) implementation
│   │   ├── baseline_lgb.py     # LightGBM classifier baseline
│   │   └── trainer.py          # Sequential data loader and training loops
│   ├── backtest/
│   │   └── simulator.py        # Latency-aware backtest simulator
│   └── dashboard/
│       └── app.py              # Streamlit dashboard layout
├── tests/
│   ├── test_book_builder.py    # Unit tests for order book updates
│   └── test_feature_extractor.py # Unit tests for feature formulas
├── run_pipeline.py             # Main pipeline orchestrator script
├── requirements.txt            # Package dependencies
└── README.md                   # Setup & usage manual
```

---

## ⚙️ Installation

1. **Set up Virtual Environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

All primary workflows are accessible via the `run_pipeline.py` script:

### 1. Run Unit Tests
Verify order book sorting and feature formulas are mathematically correct:
```powershell
python run_pipeline.py --mode test
```

### 2. Record High-Frequency Data
Stream and record live LOB depth updates from Binance spot to local CSV:
```powershell
python run_pipeline.py --mode record --symbol btcusdt --duration 300
```
This saves tick updates directly to `data/raw/btcusdt_lob_data.csv`.

### 3. Train Machine Learning Models
Orchestrate feature engineering and train model weights. If no historical CSV exists, it automatically synthesizes a demo LOB dataset:
- **LightGBM Baseline**:
  ```powershell
  python run_pipeline.py --mode train --model lgb
  ```
- **DeepLOB Model**:
  ```powershell
  python run_pipeline.py --mode train --model deeplob --epochs 10
  ```
Model checkpoints and scaler stats will be saved under `checkpoints/`.

### 4. Launch the Interactive Dashboard
Visualize live order book depth, features, predictions, and replay backtests:
```powershell
python run_pipeline.py --mode dashboard
```
Open the generated local URL (usually `http://localhost:8501`) in your browser.

---

## 📊 Backtester Logic & Latency Modeling

To prevent look-ahead bias and capture trading friction:
- **Index Shift Delay**: If execution latency is $D$ ms and tick resolution is $T$ ms, order execution is delayed by $S = \text{round}(D / T)$ ticks. A buy signal generated at tick $t$ is filled at the best ask price of tick $t + S$.
- **Fees & Slippage**: Transaction fees are computed as a fraction of traded size (e.g. 0.05%), and slippage is applied as an adverse price adjustment (e.g. 0.01% premium on buy fills, discount on sell fills).
- **Causality**: All rolling features are built using lookback windows lagging the current step, guaranteeing zero feature leakage from future states.
