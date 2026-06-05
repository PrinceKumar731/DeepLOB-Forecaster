import os
import json
import time
import argparse
import pandas as pd
import websocket
from src.features.book_builder import OrderBook
from src.features.feature_extractor import extract_basic_features

class BinanceLOBClient:
    """
    WebSocket client to fetch live Limit Order Book (LOB) data from Binance.
    Subscribes to <symbol>@depth10@100ms.
    """
    def __init__(self, symbol="btcusdt", save_dir="data/raw"):
        self.symbol = symbol.lower()
        self.save_dir = save_dir
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth10@100ms"
        
        self.book = OrderBook(symbol=symbol)
        self.is_recording = False
        self.records = []
        self.max_records_in_memory = 1000
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path = os.path.join(self.save_dir, f"{self.symbol}_lob_data.csv")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Binance depth10 message format:
            # {
            #   "lastUpdateId": 160,
            #   "bids": [["price", "qty"], ...],
            #   "asks": [["price", "qty"], ...]
            # }
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            timestamp = time.time()
            
            # Update order book state
            self.book.update_from_snapshot(bids, asks, timestamp=timestamp)
            
            if self.is_recording:
                # Get the flat feature vector (40 features for 10 levels)
                vector = self.book.get_feature_vector(k=10)
                record = {
                    "timestamp": timestamp,
                    "last_update_id": data.get("lastUpdateId")
                }
                # Add features: bid_p_0, bid_q_0, ask_p_0, ask_q_0, ...
                for i in range(10):
                    record[f"bid_p_{i}"] = vector[4 * i + 0]
                    record[f"bid_q_{i}"] = vector[4 * i + 1]
                    record[f"ask_p_{i}"] = vector[4 * i + 2]
                    record[f"ask_q_{i}"] = vector[4 * i + 3]
                    
                self.records.append(record)
                
                # Periodically flush records to disk
                if len(self.records) >= self.max_records_in_memory:
                    self.flush_records()
                    
        except Exception as e:
            print(f"Error processing message: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket Connection Closed")
        if self.is_recording:
            self.flush_records()

    def on_open(self, ws):
        print(f"Connected to Binance WebSocket for {self.symbol.upper()}")
        print(f"Subscribed to url: {self.ws_url}")

    def start(self, record=False, duration=None):
        """
        Starts the websocket client.
        record: bool, whether to save data to disk.
        duration: int, duration in seconds to run before closing.
        """
        self.is_recording = record
        if record:
            print(f"Recording active. Target save file: {self.csv_path}")
            # Write headers if file doesn't exist
            if not os.path.exists(self.csv_path):
                headers = ["timestamp", "last_update_id"]
                for i in range(10):
                    headers.extend([f"bid_p_{i}", f"bid_q_{i}", f"ask_p_{i}", f"ask_q_{i}"])
                pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)

        # websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        if duration:
            import threading
            def stop_after_duration():
                time.sleep(duration)
                print(f"Duration of {duration}s reached. Stopping WebSocket client...")
                self.ws.close()
                
            thread = threading.Thread(target=stop_after_duration)
            thread.daemon = True
            thread.start()

        self.ws.run_forever()

    def flush_records(self):
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
        print(f"Flushed {len(self.records)} records to {self.csv_path}")
        self.records.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Live LOB Data Streamer/Recorder")
    parser.add_argument("--symbol", type=str, default="btcusdt", help="Trading symbol (default: btcusdt)")
    parser.add_argument("--record", action="store_true", help="Record LOB updates to CSV")
    parser.add_argument("--duration", type=int, default=None, help="Duration in seconds to run client")
    
    args = parser.parse_args()
    
    client = BinanceLOBClient(symbol=args.symbol)
    client.start(record=args.record, duration=args.duration)
