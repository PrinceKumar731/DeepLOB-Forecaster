import React from 'react';

const Prediction = ({ prediction = "FLAT", probabilities = [0.0, 1.0, 0.0], modelName = "Rule-based OBI" }) => {
  const [probDown, probFlat, probUp] = probabilities;

  const getStyles = () => {
    switch (prediction) {
      case 'UP':
        return {
          color: 'var(--green)',
          bg: 'rgba(14, 203, 129, 0.1)',
          border: 'rgba(14, 203, 129, 0.2)',
          shadow: '0 0 15px rgba(14, 203, 129, 0.3)'
        };
      case 'DOWN':
        return {
          color: 'var(--red)',
          bg: 'rgba(246, 70, 93, 0.1)',
          border: 'rgba(246, 70, 93, 0.2)',
          shadow: '0 0 15px rgba(246, 70, 93, 0.3)'
        };
      default:
        return {
          color: 'var(--text-primary)',
          bg: 'rgba(43, 49, 57, 0.1)',
          border: 'rgba(47, 54, 63, 0.3)',
          shadow: 'none'
        };
    }
  };

  const style = getStyles();

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Model Signal</h3>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', backgroundColor: 'var(--bg-primary)', padding: '2px 8px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          {modelName}
        </span>
      </div>

      {/* Main Signal Display */}
      <div style={{
        backgroundColor: style.bg,
        border: `1px solid ${style.border}`,
        borderRadius: '8px',
        padding: '24px',
        textAlign: 'center',
        boxShadow: style.shadow,
        transition: 'all 0.3s ease-in-out'
      }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '1px' }}>
          Short-Term Horizon
        </div>
        <div style={{ fontSize: '2.5rem', fontWeight: '800', color: style.color }}>
          {prediction}
        </div>
      </div>

      {/* Probability Breakdown */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)', paddingBottom: '4px' }}>
          Probability Distribution
        </div>

        {/* Down Prob */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '3px' }}>
            <span>Price Down (Bearish)</span>
            <span style={{ fontWeight: '600' }}>{(probDown * 100).toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-primary)', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ width: `${probDown * 100}%`, height: '100%', backgroundColor: 'var(--red)', borderRadius: '3px', transition: 'width 0.3s' }} />
          </div>
        </div>

        {/* Flat Prob */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '3px' }}>
            <span>Price Flat (Neutral)</span>
            <span style={{ fontWeight: '600' }}>{(probFlat * 100).toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-primary)', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ width: `${probFlat * 100}%`, height: '100%', backgroundColor: 'var(--text-secondary)', borderRadius: '3px', transition: 'width 0.3s' }} />
          </div>
        </div>

        {/* Up Prob */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '3px' }}>
            <span>Price Up (Bullish)</span>
            <span style={{ fontWeight: '600' }}>{(probUp * 100).toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-primary)', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ width: `${probUp * 100}%`, height: '100%', backgroundColor: 'var(--green)', borderRadius: '3px', transition: 'width 0.3s' }} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Prediction;
