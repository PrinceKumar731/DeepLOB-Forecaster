import React, { useState, useEffect } from 'react';
import { Play, Loader2, DollarSign, Percent, TrendingUp } from 'lucide-react';
import PriceChart from './PriceChart';

const Backtester = ({ recordedFiles = [], modelStatus = {} }) => {
  const [selectedFile, setSelectedFile] = useState('');
  const [modelType, setModelType] = useState('rule-based');
  const [latencyMs, setLatencyMs] = useState(50);
  const [fee, setFee] = useState(0.0005);
  const [slippage, setSlippage] = useState(0.0001);
  
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [equityData, setEquityData] = useState([]);
  const [trades, setTrades] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    if (recordedFiles.length > 0 && !selectedFile) {
      setSelectedFile(recordedFiles[0]);
    }
  }, [recordedFiles, selectedFile]);

  const handleRunBacktest = async () => {
    if (!selectedFile) {
      setError('Please select a data file first.');
      return;
    }
    setError('');
    setLoading(true);
    setMetrics(null);

    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: selectedFile,
          model_type: modelType,
          latency_ms: Number(latencyMs),
          transaction_fee: Number(fee),
          slippage: Number(slippage)
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to run backtest.');
      }

      const res = await response.json();
      setMetrics(res.metrics);
      setTrades(res.trades);
      
      // format chartData for PriceChart
      setChartData(res.chart_data);
      // create separate data array for equity curve
      const formattedEquity = res.chart_data.map(item => ({
        time: item.time,
        price: item.equity
      }));
      setEquityData(formattedEquity);

    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div className="card">
        <h3 style={{ margin: '0 0 16px 0', fontSize: '1.1rem' }}>Historical Backtester</h3>
        
        {/* Settings grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
          alignItems: 'end'
        }}>
          {/* File Selector */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Historical Log File</label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              style={{
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '8px',
                color: 'var(--text-primary)',
                outline: 'none'
              }}
            >
              {recordedFiles.length === 0 && <option value="">No recorded logs found...</option>}
              {recordedFiles.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          {/* Model Type */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Model Selection</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              style={{
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '8px',
                color: 'var(--text-primary)',
                outline: 'none'
              }}
            >
              <option value="rule-based">Rule-Based OBI Threshold</option>
              <option value="lgb" disabled={!modelStatus.lgb_model_loaded}>
                LightGBM Baseline {!modelStatus.lgb_model_loaded && '(Not Trained)'}
              </option>
              <option value="deeplob" disabled={!modelStatus.deeplob_model_loaded}>
                DeepLOB CNN-LSTM {!modelStatus.deeplob_model_loaded && '(Not Trained)'}
              </option>
            </select>
          </div>

          {/* Latency MS */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Simulated Latency (ms)</label>
            <input
              type="number"
              value={latencyMs}
              onChange={(e) => setLatencyMs(e.target.value)}
              style={{
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '8px',
                color: 'var(--text-primary)',
                outline: 'none'
              }}
            />
          </div>

          {/* Fee & Slippage */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Fee & Slippage (fraction)</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="number"
                step="0.0001"
                value={fee}
                onChange={(e) => setFee(e.target.value)}
                placeholder="Fee"
                style={{
                  width: '50%',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  padding: '8px',
                  color: 'var(--text-primary)',
                  outline: 'none'
                }}
              />
              <input
                type="number"
                step="0.0001"
                value={slippage}
                onChange={(e) => setSlippage(e.target.value)}
                placeholder="Slippage"
                style={{
                  width: '50%',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  padding: '8px',
                  color: 'var(--text-primary)',
                  outline: 'none'
                }}
              />
            </div>
          </div>

          {/* Action button */}
          <button
            onClick={handleRunBacktest}
            disabled={loading}
            className="button-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              height: '38px',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
            Run Backtest
          </button>
        </div>

        {error && <div style={{ color: 'var(--red)', fontSize: '0.85rem', marginTop: '12px' }}>{error}</div>}
      </div>

      {/* Metrics Cards */}
      {metrics && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px'
        }}>
          {/* Return */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              <span>Total Return</span>
              <TrendingUp size={16} />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: metrics.total_return_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {metrics.total_return_pct.toFixed(2)}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Capital: ${(metrics.final_capital || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>

          {/* Sharpe */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              <span>Sharpe Ratio (Annualized)</span>
              <Percent size={16} />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--primary)' }}>
              {metrics.sharpe_ratio.toFixed(2)}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Risk-adjusted tick return</div>
          </div>

          {/* Max DD */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              <span>Max Drawdown</span>
              <Percent size={16} />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--red)' }}>
              {metrics.max_drawdown_pct.toFixed(2)}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Peak-to-trough decline</div>
          </div>

          {/* Trades & Win Rate */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              <span>Win Rate & Trades</span>
              <DollarSign size={16} />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>
              {metrics.win_rate_pct.toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Total Executed Trades: {metrics.total_trades}
            </div>
          </div>
        </div>
      )}

      {/* Double plot Chart Layout */}
      {metrics && chartData.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <PriceChart data={chartData} markers={trades} title="BTC Mid Price ($) & Trade Fills" />
          <PriceChart data={equityData} title="Portfolio Equity Curve ($)" />
        </div>
      )}

      {/* Trade Log */}
      {metrics && trades.length > 0 && (
        <div className="card" style={{ maxHeight: '300px', overflowY: 'auto' }}>
          <h4 style={{ margin: '0 0 12px 0' }}>Detailed Trade Log</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)' }}>
                <th style={{ padding: '8px' }}>Type</th>
                <th style={{ padding: '8px' }}>Direction</th>
                <th style={{ padding: '8px' }}>Entry Price</th>
                <th style={{ padding: '8px' }}>Exit Price</th>
                <th style={{ padding: '8px' }}>Size</th>
                <th style={{ padding: '8px' }}>Net PnL</th>
                <th style={{ padding: '8px' }}>Capital</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid rgba(47, 54, 63, 0.3)' }}>
                  <td style={{ padding: '8px', fontWeight: 'bold' }}>{t.type}</td>
                  <td style={{ padding: '8px', color: t.direction === 'Long' ? 'var(--green)' : 'var(--red)' }}>{t.direction}</td>
                  <td style={{ padding: '8px' }}>${t.entry_price ? t.entry_price.toFixed(2) : '-'}</td>
                  <td style={{ padding: '8px' }}>${t.exit_price ? t.exit_price.toFixed(2) : '-'}</td>
                  <td style={{ padding: '8px' }}>{t.qty ? t.qty.toFixed(4) : '-'}</td>
                  <td style={{ padding: '8px', fontWeight: 'bold', color: t.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {t.pnl ? `${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}` : '0.00'}
                  </td>
                  <td style={{ padding: '8px' }}>${t.capital_after ? t.capital_after.toFixed(2) : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Backtester;
