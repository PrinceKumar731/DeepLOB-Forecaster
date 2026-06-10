import React from 'react';

const OrderBook = ({ bids = [], asks = [], spread = 0, bestBid = 0, bestAsk = 0, obi = 0 }) => {
  // Sort bids descending, asks ascending (first level is best)
  // Ensure we limit to top 10 levels
  const displayBids = bids.slice(0, 10);
  const displayAsks = asks.slice(0, 10);

  // Calculate cumulative volumes for visual depth bars
  const calculateCumulative = (list) => {
    let sum = 0;
    return list.map(item => {
      sum += item[1];
      return { price: item[0], qty: item[1], cumulative: sum };
    });
  };

  const bidsWithCum = calculateCumulative(displayBids);
  const asksWithCum = calculateCumulative(displayAsks);
  
  // Find maximum cumulative volume to normalize the depth bar widths
  const maxCum = Math.max(
    bidsWithCum.length ? bidsWithCum[bidsWithCum.length - 1].cumulative : 1,
    asksWithCum.length ? asksWithCum[asksWithCum.length - 1].cumulative : 1
  );

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '400px' }}>
      <h3 style={{ margin: '0 0 12px 0', fontSize: '1.1rem', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
        Order Book
      </h3>

      {/* Column Headers */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '8px', padding: '0 4px' }}>
        <div>Price</div>
        <div style={{ textAlign: 'right' }}>Size</div>
        <div style={{ textAlign: 'right' }}>Total</div>
      </div>

      {/* Asks (Sells) - Rendered top-down in reverse order so best ask is at bottom */}
      <div style={{ flex: '1', display: 'flex', flexDirection: 'column-reverse', overflowY: 'hidden', minHeight: '140px' }}>
        {asksWithCum.map((ask, idx) => {
          const percentage = (ask.cumulative / maxCum) * 100;
          return (
            <div 
              key={`ask-${idx}`}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr 1fr',
                fontSize: '0.85rem',
                padding: '3px 4px',
                position: 'relative',
                cursor: 'pointer'
              }}
            >
              {/* Depth Background Bar */}
              <div style={{
                position: 'absolute',
                top: 0,
                right: 0,
                bottom: 0,
                width: `${percentage}%`,
                backgroundColor: 'rgba(246, 70, 93, 0.08)',
                zIndex: 0,
                transition: 'width 0.2s'
              }} />
              
              <div style={{ color: 'var(--red)', zIndex: 1 }}>{ask.price.toFixed(2)}</div>
              <div style={{ textAlign: 'right', zIndex: 1 }}>{ask.qty.toFixed(4)}</div>
              <div style={{ textAlign: 'right', color: 'var(--text-secondary)', zIndex: 1 }}>{ask.cumulative.toFixed(3)}</div>
            </div>
          );
        })}
      </div>

      {/* Spread & Mid-price Panel */}
      <div style={{
        padding: '12px 4px',
        margin: '8px 0',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        zIndex: 1
      }}>
        <div>
          <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: obi > 0 ? 'var(--green)' : obi < 0 ? 'var(--red)' : 'var(--text-primary)' }}>
            {((bestBid + bestAsk) / 2 || 0).toFixed(2)}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Mid Price
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: '500' }}>
            Spread: ${spread.toFixed(2)}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            OBI: <span style={{ color: obi >= 0 ? 'var(--green)' : 'var(--red)' }}>{obi > 0 ? `+${obi.toFixed(3)}` : obi.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Bids (Buys) - Rendered top-down in normal order (best bid at top) */}
      <div style={{ flex: '1', display: 'flex', flexDirection: 'column', overflowY: 'hidden', minHeight: '140px' }}>
        {bidsWithCum.map((bid, idx) => {
          const percentage = (bid.cumulative / maxCum) * 100;
          return (
            <div 
              key={`bid-${idx}`}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr 1fr',
                fontSize: '0.85rem',
                padding: '3px 4px',
                position: 'relative',
                cursor: 'pointer'
              }}
            >
              {/* Depth Background Bar */}
              <div style={{
                position: 'absolute',
                top: 0,
                right: 0,
                bottom: 0,
                width: `${percentage}%`,
                backgroundColor: 'rgba(14, 203, 129, 0.08)',
                zIndex: 0,
                transition: 'width 0.2s'
              }} />

              <div style={{ color: 'var(--green)', zIndex: 1 }}>{bid.price.toFixed(2)}</div>
              <div style={{ textAlign: 'right', zIndex: 1 }}>{bid.qty.toFixed(4)}</div>
              <div style={{ textAlign: 'right', color: 'var(--text-secondary)', zIndex: 1 }}>{bid.cumulative.toFixed(3)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default OrderBook;
