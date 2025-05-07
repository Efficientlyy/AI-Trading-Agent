import * as d3 from 'd3';
import { format } from 'date-fns';
import React, { useEffect, useRef, useState } from 'react';
import { SignalModel } from '../../api/tradingSignals';
import { SignalData as SignalDataType } from '../../types/signals';

// Use the SignalData type from the types directory
export type SignalData = SignalDataType;

// Update component props to accept both SignalModel and SignalData
interface SignalChartProps {
  signals: SignalModel[] | SignalData[];
  width?: number;
  height?: number;
  title?: string;
  showLegend?: boolean;
  className?: string;
  defaultSymbol?: string;
}

interface PriceDataPoint {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const SignalChart: React.FC<SignalChartProps> = ({
  signals,
  width = 800,
  height = 400,
  title = 'Trading Signals',
  showLegend = true,
  className
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    if (signals.length > 0) {
      setIsLoading(false);
      renderD3Chart();
    }
  }, [signals, width, height]);

  // D3.js chart implementation
  const renderD3Chart = () => {
    if (!svgRef.current || signals.length === 0) return;

    // Clear any existing chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Set up dimensions and margins
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom - (showLegend ? 40 : 0);

    // Create SVG with zoom capability
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create a clip path to ensure elements don't render outside the chart area
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'chart-area-clip')
      .append('rect')
      .attr('width', chartWidth)
      .attr('height', chartHeight)
      .attr('x', 0)
      .attr('y', 0);

    // Create chart group
    const chart = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create a group for all elements that should be clipped
    const chartElements = chart.append('g')
      .attr('clip-path', 'url(#chart-area-clip)');

    // Process data
    type ProcessedSignal = {
      date: Date;
      value: number;
      type: string;
      strength: number;
      source?: string;
      symbol?: string;
      timestamp: string | Date;
      id?: string;
      confidence?: number;
      description?: string;
    };

    const data: ProcessedSignal[] = signals.map(signal => {
      // Handle both SignalData and SignalModel types
      const signalType = 'type' in signal ? signal.type : signal.signal_type;
      
      return {
        ...signal,
        // Ensure the type property exists
        type: signalType,
        date: new Date(signal.timestamp),
        value: signal.strength * (signalType === 'BUY' || signalType === 'STRONG_BUY' ? 1 : -1)
      };
    }).sort((a, b) => a.date.getTime() - b.date.getTime());

    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.date) as [Date, Date])
      .range([0, chartWidth]);

    const yScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([chartHeight, 0]);

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([1, 10])  // Limit zoom scale from 1x to 10x
      .extent([[0, 0], [chartWidth, chartHeight]])
      .on('zoom', (event) => {
        // Update scales with zoom transform
        const newXScale = event.transform.rescaleX(xScale);

        // Update axes with new scales
        chart.select('.x-axis').call(
          // Type assertion to make TypeScript happy
          (g: any) => d3.axisBottom(newXScale)
            .ticks(Math.min(data.length, 10))
            .tickFormat(d => format(d as Date, 'MM/dd HH:mm'))(g)
        )
          .selectAll('text')
          .attr('transform', 'rotate(-45)')
          .style('text-anchor', 'end');

        // Update all elements that depend on the x scale
        chartElements.selectAll('.signal-point')
          .attr('cx', (d: any) => newXScale(d.date));

        chartElements.selectAll('.signal-line')
          .attr('d', d3.line<ProcessedSignal>()
            .x(d => newXScale(d.date))
            .y(d => yScale(d.value))
            .curve(d3.curveMonotoneX)(data));
      });

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(Math.min(data.length, 10))
      .tickFormat(d => format(d as Date, 'MM/dd HH:mm'));

    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => `${d}`); // Format as percentage

    // Add axes to chart
    chart.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    chart.append('g')
      .attr('class', 'y-axis')
      .call(yAxis);

    // Add axis labels
    chart.append('text')
      .attr('class', 'x-axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', chartWidth / 2)
      .attr('y', chartHeight + margin.bottom - 5)
      .text('Time');

    chart.append('text')
      .attr('class', 'y-axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90)`)
      .attr('x', -chartHeight / 2)
      .attr('y', -margin.left + 15)
      .text('Signal Strength');

    // Create color scale for signal types
    const colorScale = d3.scaleOrdinal<string>()
      .domain(['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL', 'NEUTRAL'])
      .range(['#4CAF50', '#2E7D32', '#F44336', '#B71C1C', '#9E9E9E']);

    // Add zero line
    chart.append('line')
      .attr('class', 'zero-line')
      .attr('x1', 0)
      .attr('x2', chartWidth)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#888')
      .attr('stroke-dasharray', '4')
      .attr('stroke-width', 1);

    // Add signal points
    chartElements.selectAll('.signal-point')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'signal-point')
      .attr('cx', (d: ProcessedSignal) => xScale(d.date))
      .attr('cy', (d: ProcessedSignal) => yScale(d.value))
      .attr('r', (d: ProcessedSignal) => Math.max(4, d.strength * 8))
      .attr('fill', (d: ProcessedSignal) => {
        if (d.type === 'BUY') return '#4CAF50';
        if (d.type === 'STRONG_BUY') return '#2E7D32';
        if (d.type === 'SELL') return '#F44336';
        if (d.type === 'STRONG_SELL') return '#B71C1C';
        return '#9E9E9E'; // NEUTRAL
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .attr('opacity', 0.8)
      .on('mouseover', function (event, d: ProcessedSignal) {
        d3.select(this)
          .attr('stroke-width', 2)
          .attr('opacity', 1);

        const [x, y] = d3.pointer(event);

        d3.select('.chart-tooltip')
          .style('display', 'block')
          .style('left', `${x + margin.left + 10}px`)
          .style('top', `${y + margin.top - 28}px`)
          .html(`
            <div>
              <strong>${d.type}</strong><br/>
              <span>Time: ${format(d.date, 'MM/dd/yyyy HH:mm')}</span><br/>
              <span>Strength: ${d.strength.toFixed(2)}</span>
            </div>
          `);
      })
      .on('mouseout', function () {
        d3.select(this)
          .attr('stroke-width', 1);
      });

    // Create a line generator
    const lineGenerator = d3.line<ProcessedSignal>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Add signal line
    chartElements.append('path')
      .datum(data)
      .attr('class', 'signal-line')
      .attr('fill', 'none')
      .attr('stroke', '#666')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,3')
      .attr('d', lineGenerator);

    // Create tooltip
    const tooltip = d3.select('.chart-tooltip');

    // Add tooltip interaction to signal points
    chartElements.selectAll('.signal-point')
      .on('mouseover', (event, d: any) => {
        const [x, y] = d3.pointer(event);

        tooltip
          .style('display', 'block')
          .style('left', `${x + margin.left + 10}px`)
          .style('top', `${y + margin.top - 28}px`)
          .html(`
            <div>
              <strong>${d.type}</strong><br/>
              <span>Time: ${format(d.date, 'MM/dd/yyyy HH:mm')}</span><br/>
              <span>Strength: ${d.strength.toFixed(2)}</span><br/>
              <span>Source: ${d.source || 'N/A'}</span>
            </div>
          `);
      })
      .on('mouseout', () => {
        tooltip.style('display', 'none');
      });

    // Apply zoom behavior to SVG
    svg.call(zoom as any);

    // Add zoom instructions
    svg.append('text')
      .attr('class', 'zoom-instructions')
      .attr('x', width - margin.right - 150)
      .attr('y', margin.top)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .style('fill', '#888')
      .text('Use mouse wheel to zoom');

    // Add legend if enabled
    if (showLegend) {
      const legendData = [
        { type: 'STRONG_BUY', label: 'Strong Buy' },
        { type: 'BUY', label: 'Buy' },
        { type: 'NEUTRAL', label: 'Neutral' },
        { type: 'SELL', label: 'Sell' },
        { type: 'STRONG_SELL', label: 'Strong Sell' }
      ];

      const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${margin.left},${height - 30})`);

      const legendItems = legend.selectAll('.legend-item')
        .data(legendData)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => `translate(${i * (chartWidth / legendData.length)}, 0)`);

      legendItems.append('circle')
        .attr('r', 6)
        .attr('fill', d => colorScale(d.type));

      legendItems.append('text')
        .attr('x', 10)
        .attr('y', 4)
        .text(d => d.label)
        .style('font-size', '12px');
    }
  };

  // Render chart container
  const renderChart = () => {
    return (
      <div className="d3-chart-container" style={{ width, height }}>
        <svg ref={svgRef}></svg>
        <div className="chart-tooltip" style={{ display: 'none', position: 'absolute', backgroundColor: 'rgba(0,0,0,0.8)', color: 'white', padding: '8px', borderRadius: '4px', fontSize: '12px', pointerEvents: 'none', zIndex: 100, maxWidth: '200px' }}></div>
        {signals.length === 0 && !isLoading && (
          <div className="no-data" style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
            No signal data available
          </div>
        )}
        <div className="chart-controls">
          <button
            className="reset-zoom-btn"
            onClick={() => {
              d3.select(svgRef.current)
                .transition()
                .duration(750)
                .call((d3.zoom() as any).transform, d3.zoomIdentity);
            }}
          >
            Reset Zoom
          </button>
        </div>
      </div>
    );
  };

  // Handle window resize
  const handleResize = () => {
    // In a real implementation with D3, we would redraw the chart here
    if (chartRef.current) {
      // Force a re-render by updating state
      setIsLoading(true);
      renderD3Chart();
    }
  };

  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="signal-chart-container" ref={chartRef} style={{ position: 'relative' }}>
      <div className="chart-header">
        <h3>{title}</h3>
      </div>
      {isLoading ? (
        <div className="loading-indicator">Loading signal data...</div>
      ) : (
        renderChart()
      )}
      <style>{`
        .signal-chart-container {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
          background-color: #fff;
          border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
          padding: 16px;
          margin-bottom: 20px;
        }
        .chart-header {
          margin-bottom: 16px;
          text-align: center;
        }
        .chart-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
          color: #333;
        }
        .loading-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 300px;
          color: #666;
          font-size: 16px;
        }
        .no-data {
          color: #666;
          font-size: 16px;
          text-align: center;
        }
        .d3-chart-container {
          position: relative;
        }
        :global(.x-axis path), :global(.y-axis path), :global(.x-axis line), :global(.y-axis line) {
          stroke: #ddd;
        }
        :global(.x-axis text), :global(.y-axis text) {
          fill: #666;
          font-size: 12px;
        }
        :global(.x-axis-label), :global(.y-axis-label) {
          fill: #666;
          font-size: 14px;
          font-weight: 500;
        }
        :global(.signal-point:hover) {
          cursor: pointer;
        }
        .chart-controls {
          position: absolute;
          top: 10px;
          right: 10px;
          z-index: 10;
        }
        .reset-zoom-btn {
          background-color: rgba(255, 255, 255, 0.8);
          border: 1px solid #ddd;
          border-radius: 4px;
          padding: 4px 8px;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .reset-zoom-btn:hover {
          background-color: #f0f0f0;
        }
      `}</style>
    </div>
  );
};

export default SignalChart;
