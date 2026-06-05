import numpy as np

class OrderBook:
    """
    Maintains the state of a Limit Order Book (LOB).
    Bids are sorted descending (highest price first).
    Asks are sorted ascending (lowest price first).
    """
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.last_update_id = None
        self.timestamp = None

    def update_from_snapshot(self, bids_list, asks_list, timestamp=None):
        """
        Resets and populates the order book with a complete snapshot.
        bids_list: list of [price, qty] or (price, qty)
        asks_list: list of [price, qty] or (price, qty)
        """
        self.bids = {float(price): float(qty) for price, qty in bids_list if float(qty) > 0}
        self.asks = {float(price): float(qty) for price, qty in asks_list if float(qty) > 0}
        self.timestamp = timestamp

    def update_from_delta(self, bids_delta, asks_delta, timestamp=None):
        """
        Applies incremental updates to the order book.
        A quantity of 0 removes the price level.
        """
        for price_str, qty_str in bids_delta:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for price_str, qty_str in asks_delta:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty
        
        self.timestamp = timestamp

    def get_top_k(self, k=10):
        """
        Returns the top K bid levels (sorted desc) and top K ask levels (sorted asc).
        Returns list of (price, qty) tuples.
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:k]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0], reverse=False)[:k]
        
        # If there are fewer than k levels, pad with zeros
        while len(sorted_bids) < k:
            sorted_bids.append((0.0, 0.0))
        while len(sorted_asks) < k:
            sorted_asks.append((0.0, 0.0))
            
        return sorted_bids, sorted_asks

    def get_feature_vector(self, k=10):
        """
        Returns a flat numpy array representing the top K levels of bids and asks.
        Format: [bid_price_0, bid_qty_0, ask_price_0, ask_qty_0, ... bid_price_k, bid_qty_k, ask_price_k, ask_qty_k]
        Shape: (4 * k,)
        """
        bids_k, asks_k = self.get_top_k(k)
        vector = []
        for i in range(k):
            vector.extend([bids_k[i][0], bids_k[i][1], asks_k[i][0], asks_k[i][1]])
        return np.array(vector, dtype=np.float32)

    def get_mid_price(self):
        """Returns the mid price: (best_bid + best_ask) / 2"""
        bids_k, asks_k = self.get_top_k(1)
        best_bid = bids_k[0][0]
        best_ask = asks_k[0][0]
        if best_bid == 0.0 or best_ask == 0.0:
            return None
        return (best_bid + best_ask) / 2.0

    def get_micro_price(self):
        """
        Returns the micro price: (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty)
        """
        bids_k, asks_k = self.get_top_k(1)
        best_bid, bid_qty = bids_k[0]
        best_ask, ask_qty = asks_k[0]
        if best_bid == 0.0 or best_ask == 0.0 or (bid_qty + ask_qty) == 0:
            return None
        return (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty)

    def get_spread(self):
        """Returns the absolute spread: best_ask - best_bid"""
        bids_k, asks_k = self.get_top_k(1)
        best_bid = bids_k[0][0]
        best_ask = asks_k[0][0]
        if best_bid == 0.0 or best_ask == 0.0:
            return None
        return best_ask - best_bid
