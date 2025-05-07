import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { SignalData } from '../../types/signals';

interface SignalComparisonChartProps {
  technicalSignals: SignalData[];
  sentimentSignals: SignalData[];
  combinedSignals: SignalData[];
  width?: number;
  height?: number;
}

const SignalComparisonChart: React.FC<SignalComparisonChartProps> = ({
  technicalSignals,
  sentimentSignals,
  combinedSignals,
  width = 800,
  height = 400
}) => {
  const chartRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!chartRef.current || !technicalSignals.length || !sentimentSignals.length || !combinedSignals.length) {
      return;
    }

    renderChart();
  }, [technicalSignals, sentimentSignals, combinedSignals]);

  const renderChart = () => {
    if (!chartRef.current) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll('*').remove();

    // Set up dimensions
    const margin = { top: 40, right: 30, bottom: 60, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(chartRef.current)
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Combine all signals for time domain
    const allSignals = [...technicalSignals, ...sentimentSignals, ...combinedSignals];
    
    // Parse dates
    const parseTime = d3.timeParse('%Y-%m-%dT%H:%M:%S.%LZ');
    const formatTime = d3.timeFormat('%H:%M %b %d');
    
    // Process data
    const processedData = allSignals.map(signal => ({
      ...signal,
      date: typeof signal.timestamp === 'string' 
        ? parseTime(signal.timestamp) || new Date(signal.timestamp) 
        : signal.timestamp,
      signalValue: signal.type.includes('BUY') 
        ? signal.strength 
        : signal.type.includes('SELL') 
          ? -signal.strength 
          : 0
    }));

    // Sort by date
    processedData.sort((a, b) => {
      const dateA = a.date instanceof Date ? a.date : new Date();
      const dateB = b.date instanceof Date ? b.date : new Date();
      return dateA.getTime() - dateB.getTime();
    });

    // Set up scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(processedData, d => d.date instanceof Date ? d.date : new Date()) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([innerHeight, 0]);

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(5)
      .tickFormat(d => formatTime(d as Date));
    
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => {
        if (d === 1) return 'Strong Buy';
        if (d === 0.5) return 'Buy';
        if (d === 0) return 'Neutral';
        if (d === -0.5) return 'Sell';
        if (d === -1) return 'Strong Sell';
        return '';
      });

    // Add axes to chart
    svg.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)');

    svg.append('g')
      .call(yAxis);

    // Add gridlines
    svg.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .ticks(5)
        .tickSize(-innerWidth)
        .tickFormat(() => '')
      )
      .style('stroke', '#e0e0e0')
      .style('stroke-opacity', 0.7);

    // Add a zero line
    svg.append('line')
      .attr('x1', 0)
      .attr('y1', yScale(0))
      .attr('x2', innerWidth)
      .attr('y2', yScale(0))
      .style('stroke', '#888')
      .style('stroke-width', 1)
      .style('stroke-dasharray', '4');

    // Define line generators
    const line = d3.line<any>()
      .x(d => xScale(d.date instanceof Date ? d.date : new Date()))
      .y(d => yScale(d.signalValue))
      .curve(d3.curveMonotoneX);

    // Filter data by signal source
    const technicalData = processedData.filter(d => d.source === 'TECHNICAL');
    const sentimentData = processedData.filter(d => d.source === 'SENTIMENT');
    const combinedData = processedData.filter(d => d.source === 'COMBINED');

    // Add lines
    svg.append('path')
      .datum(technicalData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6') // blue
      .attr('stroke-width', 2)
      .attr('d', line);

    svg.append('path')
      .datum(sentimentData)
      .attr('fill', 'none')
      .attr('stroke', '#8b5cf6') // purple
      .attr('stroke-width', 2)
      .attr('d', line);

    svg.append('path')
      .datum(combinedData)
      .attr('fill', 'none')
      .attr('stroke', '#10b981') // green
      .attr('stroke-width', 3)
      .attr('d', line);

    // Add points
    const addPoints = (data: any[], color: string, colorClass: string, radius: number) => {
      svg.selectAll(`.point-${colorClass}`)
        .data(data)
        .enter()
        .append('circle')
        .attr('class', `point-${colorClass}`)
        .attr('cx', d => xScale(d.date instanceof Date ? d.date : new Date()))
        .attr('cy', d => yScale(d.signalValue))
        .attr('r', radius)
        .attr('fill', color)
        .append('title')
        .text(d => `${d.type} (${d.source})\nStrength: ${(d.strength * 100).toFixed(0)}%\nTime: ${formatTime(d.date instanceof Date ? d.date : new Date())}`);
    };

    addPoints(technicalData, '#3b82f6', 'technical', 5); // blue
    addPoints(sentimentData, '#8b5cf6', 'sentiment', 5); // purple
    addPoints(combinedData, '#10b981', 'combined', 7); // green

    // Add legend
    const legend = svg.append('g')
      .attr('transform', `translate(${innerWidth - 150}, 0)`);

    const legendItems = [
      { label: 'Technical', color: '#3b82f6' },
      { label: 'Sentiment', color: '#8b5cf6' },
      { label: 'Combined', color: '#10b981' }
    ];

    legendItems.forEach((item, i) => {
      const g = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      g.append('rect')
        .attr('width', 15)
        .attr('height', 3)
        .attr('fill', item.color);

      g.append('circle')
        .attr('cx', 7.5)
        .attr('cy', 1.5)
        .attr('r', item.label === 'Combined' ? 4 : 3)
        .attr('fill', item.color);

      g.append('text')
        .attr('x', 20)
        .attr('y', 5)
        .text(item.label)
        .style('font-size', '12px')
        .attr('alignment-baseline', 'middle');
    });

    // Add title
    svg.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Signal Comparison: Technical vs Sentiment vs Combined');
  };

  return (
    <div className="signal-comparison-chart">
      <svg ref={chartRef}></svg>
    </div>
  );
};

export default SignalComparisonChart;
