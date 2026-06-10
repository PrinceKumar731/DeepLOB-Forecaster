import React from 'react';
import { Wifi, WifiOff, Cpu, RefreshCw } from 'lucide-react';

const Controls = ({ 
  connected = false, 
  activeSymbol = "BTCUSDT", 
  onSymbolChange = () => {}, 
  modelStatus = {},
  mode = "live",
  onModeChange = () => {}
}) => {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '12px 20px',
      backgroundColor: 'var(--bg-secondary)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky',
      top: 0,
      zIndex: 10
    }}>
      {/* Title logo branding */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div style={{
          backgroundColor: 'var(--primary)',
          color: '#000000',
          fontWeight: '800',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '0.9rem',
          letterSpacing: '0.5px'
        }}>
          BINANCE
        </div>
        <div style={{ fontSize: '1rem', fontWeight: 'bold', letterSpacing: '0.5px' }}>
          LOB Dynamics Forecaster
        </div>
      </div>

      {/* Mode Switches */}
      <div style={{ display: 'flex', gap: '8px', backgroundColor: 'var(--bg-primary)', padding: '3px', borderRadius: '6px', border: '1px solid var(--border)' }}>
        <button
          onClick={() => onModeChange('live')}
          style={{
            backgroundColor: mode === 'live' ? 'var(--bg-tertiary)' : 'transparent',
            border: 'none',
            color: mode === 'live' ? 'var(--text-primary)' : 'var(--text-secondary)',
            padding: '6px 16px',
            borderRadius: '4px',
            fontSize: '0.8rem',
            fontWeight: '600',
            cursor: 'pointer'
          }}
        >
          Live WebSocket
        </button>
        <button
          onClick={() => onModeChange('backtest')}
          style={{
            backgroundColor: mode === 'backtest' ? 'var(--bg-tertiary)' : 'transparent',
            border: 'none',
            color: mode === 'backtest' ? 'var(--text-primary)' : 'var(--text-secondary)',
            padding: '6px 16px',
            borderRadius: '4px',
            fontSize: '0.8rem',
            fontWeight: '600',
            cursor: 'pointer'
          }}
        >
          Replay / Backtest
        </button>
      </div>

      {/* Controls & Connection parameters */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        {/* Symbol Select (Only active in live mode) */}
        {mode === 'live' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Market:</span>
            <select
              value={activeSymbol}
              onChange={(e) => onSymbolChange(e.target.value)}
              style={{
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '6px 12px',
                color: 'var(--text-primary)',
                fontWeight: 'bold',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value="btcusdt">BTC / USDT</option>
              <option value="ethusdt">ETH / USDT</option>
              <option value="solusdt">SOL / USDT</option>
            </select>
          </div>
        )}

        {/* Connection status */}
        {mode === 'live' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: connected ? 'var(--green)' : 'var(--red)' }}>
            {connected ? (
              <>
                <Wifi size={16} />
                <span>Connected Feed</span>
              </>
            ) : (
              <>
                <WifiOff size={16} />
                <span>Feed Disconnected</span>
              </>
            )}
          </div>
        )}

        {/* Model status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderLeft: '1px solid var(--border)', paddingLeft: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: modelStatus.lgb_model_loaded ? 'var(--green)' : 'var(--text-secondary)' }}>
            <Cpu size={14} />
            <span>LGB Ready</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: modelStatus.deeplob_model_loaded ? 'var(--green)' : 'var(--text-secondary)' }}>
            <Cpu size={14} />
            <span>DeepLOB Ready</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Controls;
