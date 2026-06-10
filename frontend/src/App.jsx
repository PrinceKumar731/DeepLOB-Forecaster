import React, { useState, useEffect, useRef } from 'react';
import Controls from './components/Controls';
import OrderBook from './components/OrderBook';
import Prediction from './components/Prediction';
import PriceChart from './components/PriceChart';
import Backtester from './components/Backtester';

const App = () => {
  const [mode, setMode] = useState('live'); // 'live' or 'backtest'
  const [connected, setConnected] = useState(false);
  const [activeSymbol, setActiveSymbol] = useState('btcusdt');
  const [modelStatus, setModelStatus] = useState({ lgb_model_loaded: false, deeplob_model_loaded: false });
  const [recordedFiles, setRecordedFiles] = useState([]);
  
  // Live LOB Data
  const [liveData, setLiveData] = useState({
    bids: [],
    asks: [],
    spread: 0,
    bestBid: 0,
    bestAsk: 0,
    obi: 0,
    prediction: 'FLAT',
    probabilities: [0.15, 0.70, 0.15]
  });

  // Live price chart buffer (max 150 ticks)
  const [liveChartData, setLiveChartData] = useState([]);
  
  const wsRef = useRef(null);

  // Fetch initial statuses
  const fetchStatus = async () => {
    try {
      const res1 = await fetch('/api/status');
      if (res1.ok) {
        const data1 = await res1.json();
        setModelStatus(data1);
      }
      
      const res2 = await fetch('/api/recorded-files');
      if (res2.ok) {
        const data2 = await res2.json();
        setRecordedFiles(data2.files);
      }
    } catch (e) {
      console.warn("Failed to fetch initial status: ", e);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Poll status occasionally (every 10 seconds)
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket Live Connection management
  useEffect(() => {
    if (mode !== 'live') {
      if (wsRef.current) {
        wsRef.current.close();
      }
      return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.port === '5173' ? 'localhost:8000' : window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/ws/live`;

    console.log(`Connecting WebSocket to: ${wsUrl}`);
    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;

    socket.onopen = () => {
      setConnected(true);
      // Synchronize symbol immediately
      socket.send(JSON.stringify({ symbol: activeSymbol }));
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Update Order Book state
        setLiveData({
          bids: data.bids || [],
          asks: data.asks || [],
          spread: data.spread || 0,
          bestBid: data.best_bid || 0,
          bestAsk: data.best_ask || 0,
          obi: data.obi || 0,
          prediction: data.prediction || 'FLAT',
          probabilities: data.probabilities || [0.15, 0.70, 0.15]
        });

        // Update live price chart buffer
        const midPrice = (data.best_bid + data.best_ask) / 2;
        if (midPrice > 0) {
          setLiveChartData(prev => {
            const newTick = { time: data.timestamp, price: midPrice };
            // Append and slice
            const nextList = [...prev, newTick];
            if (nextList.length > 150) {
              return nextList.slice(nextList.length - 150);
            }
            return nextList;
          });
        }
      } catch (err) {
        console.error("Error parsing WS message: ", err);
      }
    };

    socket.onerror = (e) => {
      console.warn("WebSocket error: ", e);
      setConnected(false);
    };

    socket.onclose = () => {
      console.log("WebSocket closed.");
      setConnected(false);
      
      // Auto reconnect after 3 seconds if mode is still live
      const timer = setTimeout(() => {
        if (mode === 'live') {
          fetchStatus(); // Refresh status check
        }
      }, 3000);
      return () => clearTimeout(timer);
    };

    return () => {
      socket.close();
    };
  }, [mode, activeSymbol]);

  const handleSymbolChange = async (newSymbol) => {
    setActiveSymbol(newSymbol);
    try {
      await fetch('/api/select-symbol', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: newSymbol })
      });
      // Clear live chart buffer when switching symbols
      setLiveChartData([]);
    } catch (e) {
      console.error("Failed to select symbol: ", e);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      {/* Top sticky controls */}
      <Controls
        connected={connected}
        activeSymbol={activeSymbol}
        onSymbolChange={handleSymbolChange}
        modelStatus={modelStatus}
        mode={mode}
        onModeChange={(newMode) => {
          setMode(newMode);
          fetchStatus(); // fetch files again on switch
        }}
      />

      {/* Main Content Pane */}
      <main style={{ flex: 1, padding: '24px', maxWidth: '1400px', width: '100%', margin: '0 auto', boxSizing: 'border-box' }}>
        {mode === 'live' ? (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(12, 1fr)',
            gap: '24px',
            alignItems: 'start'
          }}>
            {/* Left side: Order book */}
            <div style={{ gridColumn: 'span 4' }}>
              <OrderBook
                bids={liveData.bids}
                asks={liveData.asks}
                spread={liveData.spread}
                bestBid={liveData.bestBid}
                bestAsk={liveData.bestAsk}
                obi={liveData.obi}
              />
            </div>

            {/* Right side: Charts and predictions */}
            <div style={{ gridColumn: 'span 8', display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <Prediction
                prediction={liveData.prediction}
                probabilities={liveData.probabilities}
                modelName="Live Classifier"
              />
              <PriceChart
                data={liveChartData}
                title={`Live Mid-Price Tick Feed (${activeSymbol.toUpperCase()})`}
              />
            </div>
          </div>
        ) : (
          <Backtester
            recordedFiles={recordedFiles}
            modelStatus={modelStatus}
          />
        )}
      </main>
    </div>
  );
};

export default App;
