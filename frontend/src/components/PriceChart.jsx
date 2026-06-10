import React, { useEffect, useRef } from 'react';
import { createChart, AreaSeries } from 'lightweight-charts';

const PriceChart = ({ data = [], markers = [], title = "Price Chart" }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create Chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 350,
      layout: {
        background: { color: '#181A20' },
        textColor: '#848E9C',
      },
      grid: {
        vertLines: { color: 'rgba(47, 54, 63, 0.15)' },
        horzLines: { color: 'rgba(47, 54, 63, 0.15)' },
      },
      timeScale: {
        borderColor: '#2F363F',
        timeVisible: false, // Hide time scale labels since we are using index numbers
      },
      rightPriceScale: {
        borderColor: '#2F363F',
        autoScale: true,
      },
    });

    // Add Area Series (Lightweight Charts v5 API)
    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: '#f0b90b',
      topColor: 'rgba(240, 185, 11, 0.15)',
      bottomColor: 'rgba(240, 185, 11, 0.0)',
      lineWidth: 2,
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });

    chartRef.current = chart;
    seriesRef.current = areaSeries;

    // Resize Handler
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Clean up
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, []);

  // Update Data and Markers in a single safe loop
  useEffect(() => {
    if (!seriesRef.current || !data || data.length === 0) return;

    try {
      // Map data array to positive integer indices to support sub-second charts safely
      const chartPoints = data.map((item, idx) => ({
        time: idx + 1, // Positive integer index
        value: item.price
      }));

      seriesRef.current.setData(chartPoints);

      // Map markers to indices
      if (markers && markers.length > 0) {
        const formattedMarkers = markers.map(m => {
          let closestIdx = -1;
          let minDiff = Infinity;
          
          for (let i = 0; i < data.length; i++) {
            const diff = Math.abs(data[i].time - m.time);
            if (diff < minDiff) {
              minDiff = diff;
              closestIdx = i;
            }
          }

          // Limit match window to 2 seconds to avoid misplaced markers
          if (closestIdx === -1 || minDiff > 2.0) return null;

          const isLong = m.direction === 'Long';
          const isEntry = m.type === 'Entry' || m.pnl === undefined;
          
          return {
            time: closestIdx + 1,
            position: isLong ? 'belowBar' : 'aboveBar',
            color: isLong ? '#0ecb81' : '#f6465d',
            shape: isLong ? 'arrowUp' : 'arrowDown',
            text: `${m.direction} ${isEntry ? 'Enter' : 'Exit'}`,
          };
        }).filter(Boolean);

        // Sort markers chronologically by index
        formattedMarkers.sort((a, b) => a.time - b.time);

        // Deduplicate markers by index (Lightweight Charts throws on duplicate times)
        const uniqueMarkers = [];
        const seenIndices = new Set();
        for (const mark of formattedMarkers) {
          if (!seenIndices.has(mark.time)) {
            seenIndices.add(mark.time);
            uniqueMarkers.push(mark);
          }
        }

        seriesRef.current.setMarkers(uniqueMarkers);
      } else {
        seriesRef.current.setMarkers([]);
      }

      // Auto fit content
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    } catch (e) {
      console.warn("Lightweight Charts render error caught: ", e);
    }
  }, [data, markers]);

  return (
    <div className="card" style={{ flex: '1', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ margin: '0 0 12px 0', fontSize: '1.1rem' }}>{title}</h3>
      <div ref={chartContainerRef} style={{ width: '100%', height: '350px' }} />
    </div>
  );
};

export default PriceChart;
