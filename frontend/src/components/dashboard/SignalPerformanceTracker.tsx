import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { format } from 'date-fns';
import { SignalData } from '../../types/signals';

interface PerformanceMetrics {
  totalSignals: number;
  correctSignals: number;
  accuracy: number;
  averageReturn: number;
  profitFactor: number;
  winRate: number;
}

interface SignalPerformance {
  signalId: string;
  signalType: string;
  entryTime: Date;
  entryPrice: number;
  exitTime: Date;
  exitPrice: number;
  return: number;
  isCorrect: boolean;
  source: string;
}

interface SignalPerformanceTrackerProps {
  symbol: string;
  timeframe: string;
  signals: SignalData[];
}

const SignalPerformanceTracker: React.FC<SignalPerformanceTrackerProps> = ({
  symbol,
  timeframe,
  signals
}) => {
  const chartRef = useRef<SVGSVGElement>(null);
  
  // Generate mock historical performance data
  const generateMockPerformanceData = (): SignalPerformance[] => {
    // Filter signals by source
    const technicalSignals = signals.filter(signal => signal.source === 'TECHNICAL');
    const sentimentSignals = signals.filter(signal => signal.source === 'SENTIMENT');
    const combinedSignals = signals.filter(signal => signal.source === 'COMBINED');
    
    // Generate performance data for each signal type
    const generatePerformance = (signalList: SignalData[], source: string): SignalPerformance[] => {
      return signalList.map(signal => {
        const entryTime = new Date(signal.timestamp);
        
        // Generate a random exit time 1-5 days after entry
        const exitTime = new Date(entryTime);
        exitTime.setDate(exitTime.getDate() + Math.floor(Math.random() * 5) + 1);
        
        // Generate random prices
        const entryPrice = 100 + Math.random() * 20;
        
        // Calculate exit price based on signal type (BUY signals should have positive returns more often)
        let returnMultiplier = 1;
        if (signal.type.includes('BUY')) {
          // 70% chance of positive return for BUY signals
          returnMultiplier = Math.random() < 0.7 ? 1 : -1;
        } else {
          // 70% chance of negative return for SELL signals (which is good)
          returnMultiplier = Math.random() < 0.7 ? -1 : 1;
        }
        
        // Generate return between -5% and +10%
        const returnPct = returnMultiplier * (Math.random() * 15 - 5) / 100;
        const exitPrice = entryPrice * (1 + returnPct);
        
        // Determine if signal was correct
        const isCorrect = (signal.type.includes('BUY') && returnPct > 0) || 
                         (signal.type.includes('SELL') && returnPct < 0);
        
        return {
          signalId: signal.id || `signal-${Math.random().toString(36).substr(2, 9)}`,
          signalType: signal.type,
          entryTime,
          entryPrice,
          exitTime,
          exitPrice,
          return: returnPct,
          isCorrect,
          source
        };
      });
    };
    
    // Combine all performance data
    return [
      ...generatePerformance(technicalSignals, 'TECHNICAL'),
      ...generatePerformance(sentimentSignals, 'SENTIMENT'),
      ...generatePerformance(combinedSignals, 'COMBINED')
    ];
  };
  
  // Calculate performance metrics
  const calculateMetrics = (performances: SignalPerformance[], source?: string): PerformanceMetrics => {
    // Filter by source if provided
    const filteredPerformances = source 
      ? performances.filter(p => p.source === source)
      : performances;
    
    if (filteredPerformances.length === 0) {
      return {
        totalSignals: 0,
        correctSignals: 0,
        accuracy: 0,
        averageReturn: 0,
        profitFactor: 0,
        winRate: 0
      };
    }
    
    const totalSignals = filteredPerformances.length;
    const correctSignals = filteredPerformances.filter(p => p.isCorrect).length;
    const accuracy = correctSignals / totalSignals;
    
    const returns = filteredPerformances.map(p => p.return);
    const averageReturn = returns.reduce((sum, r) => sum + r, 0) / totalSignals;
    
    const profits = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0);
    const losses = Math.abs(returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0));
    const profitFactor = losses === 0 ? profits : profits / losses;
    
    const winningTrades = returns.filter(r => r > 0).length;
    const winRate = winningTrades / totalSignals;
    
    return {
      totalSignals,
      correctSignals,
      accuracy,
      averageReturn,
      profitFactor,
      winRate
    };
  };
  
  // Generate performance data
  const performanceData = generateMockPerformanceData();
  
  // Calculate metrics for each signal source
  const technicalMetrics = calculateMetrics(performanceData, 'TECHNICAL');
  const sentimentMetrics = calculateMetrics(performanceData, 'SENTIMENT');
  const combinedMetrics = calculateMetrics(performanceData, 'COMBINED');
  
  // Render performance chart
  useEffect(() => {
    if (!chartRef.current || performanceData.length === 0) {
      return;
    }
    
    // Clear previous chart
    d3.select(chartRef.current).selectAll('*').remove();
    
    // Set up dimensions
    const width = 600;
    const height = 300;
    const margin = { top: 30, right: 100, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create SVG
    const svg = d3.select(chartRef.current)
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Group data by source and calculate cumulative returns
    const sourceGroups = d3.group(performanceData, d => d.source);
    
    const cumulativeReturns = Array.from(sourceGroups).map(([source, performances]) => {
      // Sort by entry time
      performances.sort((a, b) => a.entryTime.getTime() - b.entryTime.getTime());
      
      // Calculate cumulative returns
      let cumulative = 0;
      const returns = performances.map(p => {
        cumulative += p.return;
        return {
          source,
          date: p.exitTime,
          return: cumulative
        };
      });
      
      return returns;
    }).flat();
    
    // Set up scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(cumulativeReturns, d => d.date) as [Date, Date])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([
        d3.min(cumulativeReturns, d => d.return) as number * 1.1,
        d3.max(cumulativeReturns, d => d.return) as number * 1.1
      ])
      .range([innerHeight, 0]);
    
    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(5)
      .tickFormat(d => format(d as Date, 'MMM d'));
    
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => `${(d as number * 100).toFixed(1)}%`);
    
    // Add axes to chart
    svg.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis);
    
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
      .style('stroke-width', 1);
    
    // Define line generator
    const line = d3.line<any>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.return))
      .curve(d3.curveMonotoneX);
    
    // Define colors for each source
    const colors = {
      'TECHNICAL': '#3b82f6', // blue
      'SENTIMENT': '#8b5cf6', // purple
      'COMBINED': '#10b981'   // green
    };
    
    // Add lines for each source
    Array.from(sourceGroups.keys()).forEach(source => {
      const sourceData = cumulativeReturns.filter(d => d.source === source);
      
      if (sourceData.length === 0) return;
      
      // Add line
      svg.append('path')
        .datum(sourceData)
        .attr('fill', 'none')
        .attr('stroke', colors[source as keyof typeof colors])
        .attr('stroke-width', source === 'COMBINED' ? 3 : 2)
        .attr('d', line);
      
      // Add last point label
      const lastPoint = sourceData[sourceData.length - 1];
      
      svg.append('text')
        .attr('x', xScale(lastPoint.date) + 5)
        .attr('y', yScale(lastPoint.return))
        .attr('dy', '0.35em')
        .style('font-size', '12px')
        .style('fill', colors[source as keyof typeof colors])
        .text(`${source} ${(lastPoint.return * 100).toFixed(1)}%`);
    });
    
    // Add title
    svg.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text('Cumulative Returns by Signal Source');
    
    // Add x-axis label
    svg.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 40)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Date');
    
    // Add y-axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Cumulative Return');
  }, [performanceData]);
  
  return (
    <div className="signal-performance-tracker bg-white p-4 rounded-lg shadow-sm">
      <h3 className="text-lg font-medium mb-3">Signal Performance Tracker</h3>
      <p className="text-sm text-gray-600 mb-4">
        Historical performance of different signal sources for {symbol} on {timeframe} timeframe:
      </p>
      
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Technical Metrics */}
        <div className="border rounded-lg p-3 bg-blue-50">
          <h4 className="text-md font-medium text-blue-800 mb-2">Technical Signals</h4>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <p className="text-xs text-gray-600">Accuracy</p>
              <p className="text-lg font-semibold text-blue-700">
                {(technicalMetrics.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Avg Return</p>
              <p className="text-lg font-semibold text-blue-700">
                {(technicalMetrics.averageReturn * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Win Rate</p>
              <p className="text-lg font-semibold text-blue-700">
                {(technicalMetrics.winRate * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Profit Factor</p>
              <p className="text-lg font-semibold text-blue-700">
                {technicalMetrics.profitFactor.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
        
        {/* Sentiment Metrics */}
        <div className="border rounded-lg p-3 bg-purple-50">
          <h4 className="text-md font-medium text-purple-800 mb-2">Sentiment Signals</h4>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <p className="text-xs text-gray-600">Accuracy</p>
              <p className="text-lg font-semibold text-purple-700">
                {(sentimentMetrics.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Avg Return</p>
              <p className="text-lg font-semibold text-purple-700">
                {(sentimentMetrics.averageReturn * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Win Rate</p>
              <p className="text-lg font-semibold text-purple-700">
                {(sentimentMetrics.winRate * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Profit Factor</p>
              <p className="text-lg font-semibold text-purple-700">
                {sentimentMetrics.profitFactor.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
        
        {/* Combined Metrics */}
        <div className="border rounded-lg p-3 bg-green-50">
          <h4 className="text-md font-medium text-green-800 mb-2">Combined Signals</h4>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <p className="text-xs text-gray-600">Accuracy</p>
              <p className="text-lg font-semibold text-green-700">
                {(combinedMetrics.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Avg Return</p>
              <p className="text-lg font-semibold text-green-700">
                {(combinedMetrics.averageReturn * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Win Rate</p>
              <p className="text-lg font-semibold text-green-700">
                {(combinedMetrics.winRate * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Profit Factor</p>
              <p className="text-lg font-semibold text-green-700">
                {combinedMetrics.profitFactor.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Performance Chart */}
      <div className="performance-chart mb-4">
        <svg ref={chartRef}></svg>
      </div>
      
      <div className="p-3 bg-gray-50 rounded border border-gray-200">
        <h4 className="text-sm font-medium mb-1">Performance Insights</h4>
        <p className="text-xs text-gray-700">
          {combinedMetrics.averageReturn > Math.max(technicalMetrics.averageReturn, sentimentMetrics.averageReturn) ? (
            "Combined signals are outperforming both technical and sentiment signals, showing the value of our integrated approach."
          ) : combinedMetrics.averageReturn > technicalMetrics.averageReturn && combinedMetrics.averageReturn > sentimentMetrics.averageReturn ? (
            "Combined signals are performing better than individual signal sources, validating our integration strategy."
          ) : combinedMetrics.averageReturn > technicalMetrics.averageReturn ? (
            "Combined signals are outperforming technical signals but not sentiment signals. Consider increasing sentiment weight."
          ) : combinedMetrics.averageReturn > sentimentMetrics.averageReturn ? (
            "Combined signals are outperforming sentiment signals but not technical signals. Consider increasing technical weight."
          ) : (
            "Individual signal sources are currently outperforming combined signals. Consider adjusting weights or market regime detection."
          )}
        </p>
      </div>
    </div>
  );
};

export default SignalPerformanceTracker;
