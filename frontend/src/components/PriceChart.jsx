import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

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
        vertLines: { color: 'rgba(47, 54, 63, 0.3)' },
        horzLines: { color: 'rgba(47, 54, 63, 0.3)' },
      },
      timeScale: {
        borderColor: '#2F363F',
        timeVisible: true,
        secondsVisible: true,
      },
      rightPriceScale: {
        borderColor: '#2F363F',
        autoScale: true,
      },
    });

    // Add Area Series
    const areaSeries = chart.addAreaSeries({
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

  // Update Data
  useEffect(() => {
    if (!seriesRef.current || !data.length) return;

    // Format data for Lightweight Charts
    // Ensure chronological order
    const formattedData = data
      .map(item => ({
        time: item.time,
        value: item.price
      }))
      .sort((a, b) => a.time - b.time);

    // Remove duplicates on timestamp (Lightweight Charts throws errors on duplicate timestamps)
    const uniqueData = [];
    const seenTimes = new Set();
    for (const d of formattedData) {
      if (!seenTimes.has(d.time)) {
        seenTimes.add(d.time);
        uniqueData.push(d);
      }
    }

    try {
      seriesRef.current.setData(uniqueData);
      
      // Auto fit content
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    } catch (e) {
      console.warn("Failed to set chart data: ", e);
    }
  }, [data]);

  // Update Markers
  useEffect(() => {
    if (!seriesRef.current) return;

    // Format markers for Lightweight Charts
    const formattedMarkers = markers.map(m => {
      const isLong = m.direction === 'Long';
      const isEntry = m.type === 'Entry' || m.pnl === undefined;
      return {
        time: m.time,
        position: isLong ? 'belowBar' : 'aboveBar',
        color: isLong ? '#0ecb81' : '#f6465d',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        text: `${m.direction} ${isEntry ? 'Enter' : 'Exit'}`,
      };
    }).sort((a, b) => a.time - b.time);

    // Remove duplicates on timestamp for markers
    const uniqueMarkers = [];
    const seenTimes = new Set();
    for (const mark of formattedMarkers) {
      if (!seenTimes.has(mark.time)) {
        seenTimes.add(mark.time);
        uniqueMarkers.push(mark);
      }
    }

    try {
      seriesRef.current.setMarkers(uniqueMarkers);
    } catch (e) {
      console.warn("Failed to set chart markers: ", e);
    }
  }, [markers]);

  return (
    <div className="card" style={{ flex: '1', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ margin: '0 0 12px 0', fontSize: '1.1rem' }}>{title}</h3>
      <div ref={chartContainerRef} style={{ width: '100%', height: '350px' }} />
    </div>
  );
};

export default PriceChart;
