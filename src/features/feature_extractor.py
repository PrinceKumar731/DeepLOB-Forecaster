import numpy as np
import pandas as pd

def extract_basic_features(lob_states, k=10):
    """
    Computes basic tick-by-tick features from raw order book states.
    lob_states: numpy array of shape (N, 4 * k)
    Returns: pandas.DataFrame containing basic features
    """
    N = len(lob_states)
    features = {}

    # Extract prices and volumes
    p_b = []
    v_b = []
    p_a = []
    v_a = []
    
    for i in range(k):
        p_b.append(lob_states[:, 4 * i + 0])
        v_b.append(lob_states[:, 4 * i + 1])
        p_a.append(lob_states[:, 4 * i + 2])
        v_a.append(lob_states[:, 4 * i + 3])
        
    p_b = np.array(p_b) # (k, N)
    v_b = np.array(v_b) # (k, N)
    p_a = np.array(p_a) # (k, N)
    v_a = np.array(v_a) # (k, N)

    # 1. Mid-price and Micro-price
    mid_price = (p_b[0] + p_a[0]) / 2.0
    features['mid_price'] = mid_price
    
    # Avoid division by zero
    total_qty_0 = v_b[0] + v_a[0]
    total_qty_0_safe = np.where(total_qty_0 == 0, 1.0, total_qty_0)
    features['micro_price'] = (p_b[0] * v_a[0] + p_a[0] * v_b[0]) / total_qty_0_safe

    # 2. Bid-Ask Spread
    features['spread'] = p_a[0] - p_b[0]
    features['spread_relative'] = features['spread'] / np.where(mid_price == 0, 1.0, mid_price)

    # 3. Order Book Imbalance (OBI) at multiple depths
    for i in range(k):
        vol_sum = v_b[i] + v_a[i]
        vol_sum_safe = np.where(vol_sum == 0, 1.0, vol_sum)
        features[f'obi_{i}'] = (v_b[i] - v_a[i]) / vol_sum_safe

    # 4. Depth-Weighted Volatility and Price
    # Calculate depth-weighted price over K levels: Sum(P_i * V_i) / Sum(V_i)
    total_bid_qty = np.sum(v_b, axis=0)
    total_ask_qty = np.sum(v_a, axis=0)
    total_bid_qty_safe = np.where(total_bid_qty == 0, 1.0, total_bid_qty)
    total_ask_qty_safe = np.where(total_ask_qty == 0, 1.0, total_ask_qty)

    weighted_bid_price = np.sum(p_b * v_b, axis=0) / total_bid_qty_safe
    weighted_ask_price = np.sum(p_a * v_a, axis=0) / total_ask_qty_safe
    features['weighted_mid'] = (weighted_bid_price + weighted_ask_price) / 2.0

    # 5. Order Flow Imbalance (OFI) at level 0
    # OFI_t = dV_bid_t - dV_ask_t
    df_p_b0 = np.diff(p_b[0], prepend=p_b[0, 0])
    df_p_a0 = np.diff(p_a[0], prepend=p_a[0, 0])
    
    dv_bid = np.zeros(N)
    dv_ask = np.zeros(N)
    
    # t=0
    dv_bid[0] = 0
    dv_ask[0] = 0
    
    # Vectorized OFI calculation
    # Bid side
    # Case 1: p_b_t > p_b_{t-1} => dv_bid = v_b_t
    dv_bid = np.where(df_p_b0 > 0, v_b[0], dv_bid)
    # Case 2: p_b_t == p_b_{t-1} => dv_bid = v_b_t - v_b_{t-1}
    v_b_diff = np.diff(v_b[0], prepend=v_b[0, 0])
    dv_bid = np.where(df_p_b0 == 0, v_b_diff, dv_bid)
    # Case 3: p_b_t < p_b_{t-1} => dv_bid = 0
    dv_bid = np.where(df_p_b0 < 0, 0.0, dv_bid)

    # Ask side
    # Case 1: p_a_t < p_a_{t-1} => dv_ask = v_a_t
    dv_ask = np.where(df_p_a0 < 0, v_a[0], dv_ask)
    # Case 2: p_a_t == p_a_{t-1} => dv_ask = v_a_t - v_a_{t-1}
    v_a_diff = np.diff(v_a[0], prepend=v_a[0, 0])
    dv_ask = np.where(df_p_a0 == 0, v_a_diff, dv_ask)
    # Case 3: p_a_t > p_a_{t-1} => dv_ask = 0
    dv_ask = np.where(df_p_a0 > 0, 0.0, dv_ask)

    features['ofi_0'] = dv_bid - dv_ask

    return pd.DataFrame(features)

def add_rolling_features(df, windows=[10, 50, 100]):
    """
    Computes rolling averages and volatility on basic features to capture temporal history.
    """
    new_features = {}
    
    # We track volatility of micro_price, spread, and OBI
    for w in windows:
        # Micro price return volatility (proxy for high frequency volatility)
        returns = df['micro_price'].pct_change().fillna(0)
        new_features[f'volatility_{w}'] = returns.rolling(window=w, min_periods=1).std().fillna(0)
        
        # Rolling mean of Spread
        new_features[f'spread_mean_{w}'] = df['spread'].rolling(window=w, min_periods=1).mean()
        
        # Rolling mean of OBI level 0
        new_features[f'obi_0_mean_{w}'] = df['obi_0'].rolling(window=w, min_periods=1).mean()
        
        # Rolling sum of OFI (net buying pressure over window)
        new_features[f'ofi_sum_{w}'] = df['ofi_0'].rolling(window=w, min_periods=1).sum()

    for col, series in new_features.items():
        df[col] = series
        
    return df

def generate_labels(mid_prices, horizon=10, threshold=0.0002):
    """
    Generates labels for classification (DeepLOB style).
    - 2: Price increase (Up)
    - 1: Flat
    - 0: Price decrease (Down)
    
    Formula:
    m_minus = mean(mid_prices[t])
    m_plus = mean(mid_prices[t+1 : t+horizon])
    change = (m_plus - m_minus) / m_minus
    
    label = 2 if change > threshold
    label = 0 if change < -threshold
    label = 1 otherwise
    """
    N = len(mid_prices)
    labels = np.ones(N, dtype=np.int64) # Default to 1 (Flat)
    
    # Rolling mean for future prices
    # We use a moving average of future mid-prices to filter noise
    future_means = np.zeros(N)
    for i in range(N):
        end_idx = min(i + horizon, N)
        if end_idx > i:
            future_means[i] = np.mean(mid_prices[i:end_idx])
        else:
            future_means[i] = mid_prices[i]
            
    # Calculate percentage change
    pct_changes = (future_means - mid_prices) / np.where(mid_prices == 0, 1.0, mid_prices)
    
    labels[pct_changes > threshold] = 2
    labels[pct_changes < -threshold] = 0
    
    # Mask out the last few samples where we don't have a full prediction horizon
    labels[N - horizon:] = -1 # Sentinel value to ignore in training
    
    return labels
