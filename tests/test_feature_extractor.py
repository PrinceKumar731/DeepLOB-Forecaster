import pytest
import numpy as np
import pandas as pd
from src.features.feature_extractor import extract_basic_features, add_rolling_features, generate_labels

def test_extract_basic_features():
    # Construct a dummy lob_states of shape (2, 40) representing 2 timesteps for 10 levels
    # Column format: p_b_0, v_b_0, p_a_0, v_a_0, ...
    state_1 = np.zeros(40)
    # Lvl 0: Bid 100.0 (qty 1.0), Ask 101.0 (qty 3.0)
    state_1[0] = 100.0; state_1[1] = 1.0; state_1[2] = 101.0; state_1[3] = 3.0
    # Lvl 1: Bid 99.0 (qty 2.0), Ask 102.0 (qty 4.0)
    state_1[4] = 99.0; state_1[5] = 2.0; state_1[6] = 102.0; state_1[7] = 4.0
    
    state_2 = np.zeros(40)
    # Lvl 0: Bid 100.5 (qty 2.0), Ask 101.5 (qty 2.0)
    state_2[0] = 100.5; state_2[1] = 2.0; state_2[2] = 101.5; state_2[3] = 2.0
    
    lob_states = np.vstack([state_1, state_2])
    
    df = extract_basic_features(lob_states, k=10)
    
    assert len(df) == 2
    # Timestep 1 mid = 100.5, micro = (100*3 + 101*1)/4 = 401/4 = 100.25
    assert df['mid_price'].iloc[0] == 100.5
    assert df['micro_price'].iloc[0] == 100.25
    assert df['spread'].iloc[0] == 1.0
    assert df['obi_0'].iloc[0] == (1.0 - 3.0) / (1.0 + 3.0) # -0.5
    
    # Timestep 2 mid = 101.0, micro = (100.5*2 + 101.5*2)/4 = 101.0
    assert df['mid_price'].iloc[1] == 101.0
    assert df['micro_price'].iloc[1] == 101.0
    assert df['spread'].iloc[1] == 1.0
    assert df['obi_0'].iloc[1] == 0.0

def test_generate_labels():
    mid_prices = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.5, 100.4, 100.3])
    labels = generate_labels(mid_prices, horizon=3, threshold=0.001)
    
    # Check shape
    assert len(labels) == len(mid_prices)
    # Check that tail is flagged as -1 (since horizon goes past end)
    assert labels[-1] == -1
    assert labels[-2] == -1
    assert labels[-3] == -1
