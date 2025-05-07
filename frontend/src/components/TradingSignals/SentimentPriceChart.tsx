import React, { useState, useEffect, useRef, useCallback } from 'react';
import { format } from 'date-fns';
import * as d3 from 'd3';

interface SentimentData {
  timestamp: string | Date;
  sentiment_score: number;
  source: string;
  symbol?: string;
  article_count?: number;
}

interface PriceData {
  timestamp: string | Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface SentimentPriceChartProps {
  sentimentData: SentimentData[];
  priceData: PriceData[];
  width?: number;
  height?: number;
  title?: string;
  symbol?: string;
  timeframe?: string;
  className?: string;
}

const SentimentPriceChart: React.FC<SentimentPriceChartProps> = ({
  sentimentData,
  priceData,
  width = 900,
  height = 500,
  title = 'Sentiment & Price Analysis',
  symbol = 'BTC',
  timeframe = '1D',
  className
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>(timeframe);
  const [hoveredData, setHoveredData] = useState<any>(null);

  // Process data for visualization
  const processData = useCallback(() => {
    // Convert timestamps to Date objects
    const processedSentiment = sentimentData.map(item => ({
      ...item,
      timestamp: new Date(item.timestamp)
    })).sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    const processedPrice = priceData.map(item => ({
      ...item,
      timestamp: new Date(item.timestamp)
    })).sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    // Filter data based on selected timeframe
    const now = new Date();
    let startDate = new Date();
    
    switch (selectedTimeframe) {
      case '1D':
        startDate.setDate(now.getDate() - 1);
        break;
      case '1W':
        startDate.setDate(now.getDate() - 7);
        break;
      case '1M':
        startDate.setMonth(now.getMonth() - 1);
        break;
      case '3M':
        startDate.setMonth(now.getMonth() - 3);
        break;
      case '6M':
        startDate.setMonth(now.getMonth() - 6);
        break;
      case '1Y':
        startDate.setFullYear(now.getFullYear() - 1);
        break;
      case 'ALL':
        // No filtering
        break;
      default:
        startDate.setDate(now.getDate() - 7); // Default to 1 week
    }

    const filteredSentiment = selectedTimeframe === 'ALL' 
      ? processedSentiment 
      : processedSentiment.filter(item => item.timestamp >= startDate);
    
    const filteredPrice = selectedTimeframe === 'ALL' 
      ? processedPrice 
      : processedPrice.filter(item => item.timestamp >= startDate);

    return { sentiment: filteredSentiment, price: filteredPrice };
  }, [sentimentData, priceData, selectedTimeframe]);

  // D3.js chart implementation
  const renderD3Chart = useCallback(() => {
    if (!svgRef.current || sentimentData.length === 0 || priceData.length === 0) return;

    // Clear any existing chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Process data
    const { sentiment, price } = processData();

    // Set up dimensions and margins
    const margin = { top: 40, right: 80, bottom: 60, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // Price chart height will be 70% of total height, sentiment 30%
    const priceChartHeight = chartHeight * 0.7;
    const sentimentChartHeight = chartHeight * 0.3;

    // Create SVG with zoom capability
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create clip paths
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'price-chart-clip')
      .append('rect')
      .attr('width', chartWidth)
      .attr('height', priceChartHeight)
      .attr('x', 0)
      .attr('y', 0);

    svg.append('defs')
      .append('clipPath')
      .attr('id', 'sentiment-chart-clip')
      .append('rect')
      .attr('width', chartWidth)
      .attr('height', sentimentChartHeight)
      .attr('x', 0)
      .attr('y', 0);

    // Create chart groups
    const priceChart = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .attr('class', 'price-chart');

    const sentimentChart = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top + priceChartHeight})`)
      .attr('class', 'sentiment-chart');

    // Create clipped chart elements
    const priceChartElements = priceChart.append('g')
      .attr('clip-path', 'url(#price-chart-clip)');

    const sentimentChartElements = sentimentChart.append('g')
      .attr('clip-path', 'url(#sentiment-chart-clip)');

    // Create scales
    // X scale (shared between both charts)
    const xScale = d3.scaleTime()
      .domain([
        d3.min(price, d => d.timestamp) || new Date(),
        d3.max(price, d => d.timestamp) || new Date()
      ])
      .range([0, chartWidth]);

    // Y scale for price
    const yPriceScale = d3.scaleLinear()
      .domain([
        (d3.min(price, d => d.low) || 0) * 0.99,
        (d3.max(price, d => d.high) || 0) * 1.01
      ])
      .range([priceChartHeight, 0]);

    // Y scale for sentiment
    const ySentimentScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([sentimentChartHeight, 0]);

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(Math.min(price.length, 10))
      .tickFormat(d => format(d as Date, 'MM/dd HH:mm'));

    const yPriceAxis = d3.axisLeft(yPriceScale)
      .ticks(5)
      .tickFormat(d => `$${d}`);

    const yPriceAxisRight = d3.axisRight(yPriceScale)
      .ticks(5)
      .tickFormat(d => `$${d}`);

    const ySentimentAxis = d3.axisLeft(ySentimentScale)
      .ticks(5)
      .tickFormat(d => d.toString());

    // Add axes to chart
    priceChart.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${priceChartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    priceChart.append('g')
      .attr('class', 'y-axis')
      .call(yPriceAxis);

    priceChart.append('g')
      .attr('class', 'y-axis-right')
      .attr('transform', `translate(${chartWidth},0)`)
      .call(yPriceAxisRight);

    sentimentChart.append('g')
      .attr('class', 'y-axis')
      .call(ySentimentAxis);

    sentimentChart.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${sentimentChartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    // Add axis labels
    svg.append('text')
      .attr('class', 'x-axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', margin.left + chartWidth / 2)
      .attr('y', height - 5)
      .text('Time');

    svg.append('text')
      .attr('class', 'y-axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + priceChartHeight / 2))
      .attr('y', margin.left / 3)
      .text('Price');

    svg.append('text')
      .attr('class', 'y-axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + priceChartHeight + sentimentChartHeight / 2))
      .attr('y', margin.left / 3)
      .text('Sentiment');

    // Add title
    svg.append('text')
      .attr('class', 'chart-title')
      .attr('text-anchor', 'middle')
      .attr('x', margin.left + chartWidth / 2)
      .attr('y', margin.top / 2)
      .text(`${title} - ${symbol}`);

    // Create candlestick chart
    priceChartElements.selectAll('.candlestick')
      .data(price)
      .enter()
      .append('g')
      .attr('class', 'candlestick')
      .each(function(d: any) {
        const g = d3.select(this);
        
        // Draw the wick (high-low line)
        g.append('line')
          .attr('class', 'wick')
          .attr('x1', xScale(d.timestamp))
          .attr('x2', xScale(d.timestamp))
          .attr('y1', yPriceScale(d.high))
          .attr('y2', yPriceScale(d.low))
          .attr('stroke', d.close > d.open ? '#4CAF50' : '#F44336')
          .attr('stroke-width', 1);
        
        // Draw the body (open-close rectangle)
        const bodyHeight = Math.max(1, Math.abs(yPriceScale(d.open) - yPriceScale(d.close)));
        g.append('rect')
          .attr('class', 'body')
          .attr('x', xScale(d.timestamp) - 3)
          .attr('y', yPriceScale(Math.max(d.open, d.close)))
          .attr('width', 6)
          .attr('height', bodyHeight)
          .attr('fill', d.close > d.open ? '#4CAF50' : '#F44336');
      });

    // Create sentiment line
    const sentimentLine = d3.line<any>()
      .x(d => xScale(d.timestamp))
      .y(d => ySentimentScale(d.sentiment_score))
      .curve(d3.curveMonotoneX);

    sentimentChartElements.append('path')
      .datum(sentiment)
      .attr('class', 'sentiment-line')
      .attr('fill', 'none')
      .attr('stroke', '#2196F3')
      .attr('stroke-width', 2)
      .attr('d', sentimentLine);

    // Add zero line for sentiment
    sentimentChartElements.append('line')
      .attr('class', 'zero-line')
      .attr('x1', 0)
      .attr('x2', chartWidth)
      .attr('y1', ySentimentScale(0))
      .attr('y2', ySentimentScale(0))
      .attr('stroke', '#888')
      .attr('stroke-dasharray', '4')
      .attr('stroke-width', 1);

    // Add sentiment points
    sentimentChartElements.selectAll('.sentiment-point')
      .data(sentiment)
      .enter()
      .append('circle')
      .attr('class', 'sentiment-point')
      .attr('cx', d => xScale(d.timestamp))
      .attr('cy', d => ySentimentScale(d.sentiment_score))
      .attr('r', (d: any) => Math.max(3, (d.article_count || 1) * 0.5))
      .attr('fill', d => d.sentiment_score > 0 ? '#4CAF50' : '#F44336')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .attr('opacity', 0.7);

    // Create tooltip
    const tooltip = d3.select('.chart-tooltip');

    // Add tooltip interaction to sentiment points
    sentimentChartElements.selectAll('.sentiment-point')
      .on('mouseover', function(event, d: any) {
        d3.select(this)
          .attr('r', (d: any) => Math.max(5, (d.article_count || 1) * 0.7))
          .attr('opacity', 1);

        const [x, y] = d3.pointer(event);
        
        tooltip
          .style('display', 'block')
          .style('left', `${x + margin.left + 10}px`)
          .style('top', `${y + margin.top + priceChartHeight - 28}px`)
          .html(`
            <div>
              <strong>Sentiment: ${d.sentiment_score.toFixed(2)}</strong><br/>
              <span>Time: ${format(d.timestamp, 'MM/dd/yyyy HH:mm')}</span><br/>
              <span>Articles: ${d.article_count || 1}</span><br/>
              <span>Source: ${d.source}</span>
            </div>
          `);
          
        // Update hover state
        setHoveredData({
          timestamp: d.timestamp,
          sentiment: d.sentiment_score,
          source: d.source,
          articles: d.article_count || 1
        });
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('r', (d: any) => Math.max(3, (d.article_count || 1) * 0.5))
          .attr('opacity', 0.7);
          
        tooltip.style('display', 'none');
        setHoveredData(null);
      });

    // Add tooltip interaction to candlesticks
    priceChartElements.selectAll('.candlestick')
      .on('mouseover', function(event, d: any) {
        d3.select(this).select('.body')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);

        const [x, y] = d3.pointer(event);
        
        tooltip
          .style('display', 'block')
          .style('left', `${x + margin.left + 10}px`)
          .style('top', `${y + margin.top - 28}px`)
          .html(`
            <div>
              <strong>${symbol}</strong><br/>
              <span>Time: ${format(d.timestamp, 'MM/dd/yyyy HH:mm')}</span><br/>
              <span>Open: $${d.open.toFixed(2)}</span><br/>
              <span>High: $${d.high.toFixed(2)}</span><br/>
              <span>Low: $${d.low.toFixed(2)}</span><br/>
              <span>Close: $${d.close.toFixed(2)}</span><br/>
              <span>Volume: ${d.volume.toLocaleString()}</span>
            </div>
          `);
      })
      .on('mouseout', function() {
        d3.select(this).select('.body')
          .attr('stroke', 'none');
          
        tooltip.style('display', 'none');
      });

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([1, 10])
      .extent([[0, 0], [chartWidth, chartHeight]])
      .on('zoom', (event) => {
        // Update scales with zoom transform
        const newXScale = event.transform.rescaleX(xScale);
        
        // Update axes
        priceChart.select('.x-axis').call(
          (g: any) => d3.axisBottom(newXScale)
            .ticks(Math.min(price.length, 10))
            .tickFormat(d => format(d as Date, 'MM/dd HH:mm'))(g)
        );
        
        sentimentChart.select('.x-axis').call(
          (g: any) => d3.axisBottom(newXScale)
            .ticks(Math.min(price.length, 10))
            .tickFormat(d => format(d as Date, 'MM/dd HH:mm'))(g)
        );
        
        // Update candlesticks
        priceChartElements.selectAll('.candlestick').each(function(d: any) {
          const g = d3.select(this);
          
          g.select('.wick')
            .attr('x1', newXScale(d.timestamp))
            .attr('x2', newXScale(d.timestamp));
            
          g.select('.body')
            .attr('x', newXScale(d.timestamp) - 3);
        });
        
        // Update sentiment line
        const newSentimentLine = d3.line<any>()
          .x(d => newXScale(d.timestamp))
          .y(d => ySentimentScale(d.sentiment_score))
          .curve(d3.curveMonotoneX);
          
        sentimentChartElements.select('.sentiment-line')
          .attr('d', newSentimentLine(sentiment));
          
        // Update sentiment points
        sentimentChartElements.selectAll('.sentiment-point')
          .attr('cx', (d: any) => newXScale(d.timestamp));
      });

    // Apply zoom behavior to SVG
    svg.call(zoom as any);
  }, [sentimentData, priceData, width, height, selectedTimeframe, symbol, title, processData]);

  useEffect(() => {
    if (sentimentData.length > 0 && priceData.length > 0) {
      setIsLoading(false);
      renderD3Chart();
    }
  }, [sentimentData, priceData, renderD3Chart]);

  // Render chart container
  const renderChart = () => {
    return (
      <div ref={chartRef} className="relative">
        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-80 z-10">
            <div className="spinner-border text-primary" role="status">
              <span className="sr-only">Loading...</span>
            </div>
          </div>
        )}
        
        {/* SVG Container */}
        <svg ref={svgRef} />
        
        {/* Tooltip */}
        <div className="chart-tooltip absolute bg-gray-900 text-white p-2 rounded text-sm shadow-lg hidden pointer-events-none z-20" />
        
        {/* Timeframe selector */}
        <div className="timeframe-selector absolute top-2 right-2 flex space-x-1 z-10">
          {['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'].map(tf => (
            <button
              key={tf}
              className={`px-2 py-1 text-xs rounded ${selectedTimeframe === tf ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
              onClick={() => setSelectedTimeframe(tf)}
            >
              {tf}
            </button>
          ))}
        </div>
        
        {/* Hover info panel */}
        {hoveredData && (
          <div className="hover-info absolute bottom-2 left-2 bg-white bg-opacity-90 p-2 rounded shadow-md z-10">
            <div className="text-sm font-bold">
              {format(hoveredData.timestamp, 'yyyy-MM-dd HH:mm')}
            </div>
            <div className="text-sm">
              Sentiment: <span className={hoveredData.sentiment > 0 ? 'text-green-600' : 'text-red-600'}>
                {hoveredData.sentiment.toFixed(2)}
              </span>
            </div>
            <div className="text-xs text-gray-600">
              Articles: {hoveredData.articles} | Source: {hoveredData.source}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`sentiment-price-chart ${className || ''}`}>
      <h3 className="text-lg font-bold mb-2">{title} - {symbol}</h3>
      {renderChart()}
    </div>
  );
};

export default SentimentPriceChart;
