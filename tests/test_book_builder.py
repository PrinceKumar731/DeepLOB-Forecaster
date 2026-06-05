import pytest
import numpy as np
from src.features.book_builder import OrderBook

def test_order_book_snapshot():
    ob = OrderBook(symbol="BTCUSDT")
    bids = [["100.0", "1.5"], ["99.5", "2.0"], ["101.0", "0.5"]] # note: unsorted input
    asks = [["102.0", "1.0"], ["103.0", "2.5"], ["101.5", "0.8"]]
    
    ob.update_from_snapshot(bids, asks)
    
    # Check sorting
    sorted_bids, sorted_asks = ob.get_top_k(k=3)
    
    # Bids should be sorted desc: 101.0, 100.0, 99.5
    assert sorted_bids[0][0] == 101.0
    assert sorted_bids[1][0] == 100.0
    assert sorted_bids[2][0] == 99.5
    
    # Asks should be sorted asc: 101.5, 102.0, 103.0
    assert sorted_asks[0][0] == 101.5
    assert sorted_asks[1][0] == 102.0
    assert sorted_asks[2][0] == 103.0

def test_order_book_delta():
    ob = OrderBook(symbol="BTCUSDT")
    bids = [["100.0", "1.5"], ["99.0", "2.0"]]
    asks = [["101.0", "1.0"], ["102.0", "2.5"]]
    ob.update_from_snapshot(bids, asks)
    
    # Apply deltas
    # Update bid 100.0 qty, add bid 98.0, delete bid 99.0
    ob.update_from_delta(
        bids_delta=[["100.0", "1.8"], ["98.0", "3.0"], ["99.0", "0.0"]],
        asks_delta=[["101.0", "0.0"], ["103.0", "4.0"]] # remove ask 101.0, add ask 103.0
    )
    
    sorted_bids, sorted_asks = ob.get_top_k(k=2)
    
    assert sorted_bids[0] == (100.0, 1.8)
    assert sorted_bids[1] == (98.0, 3.0)
    
    assert sorted_asks[0] == (102.0, 2.5)
    assert sorted_asks[1] == (103.0, 4.0)

def test_feature_vector_padding():
    ob = OrderBook(symbol="BTCUSDT")
    bids = [["100.0", "1.5"]]
    asks = [["101.0", "1.0"]]
    ob.update_from_snapshot(bids, asks)
    
    vector = ob.get_feature_vector(k=3)
    assert len(vector) == 12
    # Check level 0
    assert vector[0] == 100.0
    assert vector[1] == 1.5
    assert vector[2] == 101.0
    assert vector[3] == 1.0
    # Check padded levels
    assert vector[4] == 0.0
    assert vector[5] == 0.0
    assert vector[8] == 0.0
    assert vector[9] == 0.0
