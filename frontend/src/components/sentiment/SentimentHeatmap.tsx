import React, { useMemo } from 'react';
import { format } from 'date-fns';

interface SentimentData {
  symbol: string;
  sentiment: number;
  confidence?: number;
  timestamp: string;
  volume?: number;
}

interface SentimentHeatmapProps {
  data: SentimentData[];
  width?: string;
  height?: number;
  title?: string;
  showLabels?: boolean;
}

/**
 * Component that visualizes sentiment as a heatmap across multiple assets
 */
const SentimentHeatmap: React.FC<SentimentHeatmapProps> = ({
  data,
  width = '100%',
  height = 400,
  title = 'Sentiment Heatmap',
  showLabels = true
}) => {
  // Group data by symbol and prepare for rendering
  const groupedData = useMemo(() => {
    // Group by symbol
    const groups: Record<string, SentimentData[]> = {};
    data.forEach(item => {
      if (!groups[item.symbol]) {
        groups[item.symbol] = [];
      }
      groups[item.symbol].push(item);
    });

    // Sort each group by timestamp (newest first)
    Object.keys(groups).forEach(symbol => {
      groups[symbol].sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    });

    return groups;
  }, [data]);

  // Get sentiment color based on the sentiment value
  const getSentimentColor = (sentiment: number, alpha: number = 1) => {
    if (sentiment > 0) {
      // Green for positive sentiment (darker green for stronger sentiment)
      const intensity = Math.min(1, sentiment * 2); // Scale to 0-1
      return `rgba(46, ${180 + Math.floor(intensity * 30)}, 46, ${alpha})`;
    } else if (sentiment < 0) {
      // Red for negative sentiment (darker red for stronger sentiment)
      const intensity = Math.min(1, -sentiment * 2); // Scale to 0-1
      return `rgba(${180 + Math.floor(intensity * 30)}, 46, 46, ${alpha})`;
    }
    // Grey for neutral
    return `rgba(150, 150, 150, ${alpha})`;
  };

  // Get a tooltip text for a cell
  const getCellTooltip = (item: SentimentData) => {
    const formattedTimestamp = format(new Date(item.timestamp), 'MMM d, yyyy HH:mm');
    const sentimentText = item.sentiment > 0.3 ? 'Positive' : 
                           item.sentiment < -0.3 ? 'Negative' : 'Neutral';
    const confidenceText = item.confidence 
      ? `Confidence: ${(item.confidence * 100).toFixed(0)}%` 
      : '';
    const volumeText = item.volume 
      ? `Volume: ${item.volume.toLocaleString()}` 
      : '';

    return `${item.symbol}: ${sentimentText} (${item.sentiment.toFixed(2)})
${formattedTimestamp}
${confidenceText}
${volumeText}`;
  };

  // Sort symbols by average sentiment (most positive to most negative)
  const sortedSymbols = useMemo(() => {
    return Object.keys(groupedData).sort((a, b) => {
      const aAvg = groupedData[a].reduce((sum, item) => sum + item.sentiment, 0) / groupedData[a].length;
      const bAvg = groupedData[b].reduce((sum, item) => sum + item.sentiment, 0) / groupedData[b].length;
      return bAvg - aAvg; // Most positive first
    });
  }, [groupedData]);

  // Determine cell size based on container dimensions and data
  const numSymbols = sortedSymbols.length;
  const maxCells = Math.max(...Object.values(groupedData).map(items => items.length));
  
  // Calculate dimensions for the grid
  const cellWidth = `${100 / maxCells}%`;
  const cellHeight = Math.floor(height / Math.max(numSymbols, 1));

  return (
    <div style={{ width }}>
      <h3 className="text-lg font-medium mb-2 text-white">{title}</h3>
      <div className="sentiment-heatmap" style={{ 
        width: '100%',
        height: cellHeight * numSymbols,
        display: 'flex',
        flexDirection: 'column'
      }}>
        {sortedSymbols.map((symbol, symbolIndex) => {
          const items = groupedData[symbol];
          
          return (
            <div 
              key={symbol}
              className="heatmap-row"
              style={{
                display: 'flex',
                height: cellHeight,
                width: '100%',
                position: 'relative'
              }}
            >
              {/* Symbol label */}
              {showLabels && (
                <div 
                  className="symbol-label"
                  style={{
                    width: '80px',
                    height: cellHeight,
                    display: 'flex',
                    alignItems: 'center',
                    paddingRight: '10px',
                    fontWeight: 'bold',
                    position: 'sticky',
                    left: 0,
                    backgroundColor: '#1f2937',
                    color: 'white',
                    zIndex: 1
                  }}
                >
                  {symbol}
                </div>
              )}
              
              {/* Sentiment cells */}
              <div className="cells-container" style={{ 
                flex: 1, 
                display: 'flex',
                height: cellHeight
              }}>
                {items.map((item, i) => (
                  <div
                    key={i}
                    className="sentiment-cell"
                    style={{
                      width: cellWidth,
                      height: cellHeight - 4,
                      backgroundColor: getSentimentColor(item.sentiment),
                      margin: '2px',
                      position: 'relative',
                      borderRadius: '3px',
                      cursor: 'pointer'
                    }}
                    title={getCellTooltip(item)}
                  >
                    {/* Optional cell content */}
                    <div 
                      className="cell-content"
                      style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        fontSize: '10px',
                        color: Math.abs(item.sentiment) > 0.4 ? 'white' : 'rgba(255, 255, 255, 0.8)',
                        fontWeight: 'bold'
                      }}
                    >
                      {item.sentiment.toFixed(1)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
        
        {/* Legend */}
        <div className="heatmap-legend" style={{ 
          display: 'flex',
          justifyContent: 'center',
          marginTop: '10px',
          padding: '5px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ 
              width: '20px', 
              height: '20px', 
              backgroundColor: getSentimentColor(-1),
              marginRight: '5px',
              borderRadius: '3px'
            }}></div>
            <span style={{ marginRight: '15px', color: 'rgba(255, 255, 255, 0.8)' }}>Very Negative</span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ 
              width: '20px', 
              height: '20px', 
              backgroundColor: getSentimentColor(-0.5),
              marginRight: '5px',
              borderRadius: '3px'
            }}></div>
            <span style={{ marginRight: '15px', color: 'rgba(255, 255, 255, 0.8)' }}>Negative</span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ 
              width: '20px', 
              height: '20px', 
              backgroundColor: getSentimentColor(0),
              marginRight: '5px',
              borderRadius: '3px'
            }}></div>
            <span style={{ marginRight: '15px', color: 'rgba(255, 255, 255, 0.8)' }}>Neutral</span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ 
              width: '20px', 
              height: '20px', 
              backgroundColor: getSentimentColor(0.5),
              marginRight: '5px',
              borderRadius: '3px'
            }}></div>
            <span style={{ marginRight: '15px', color: 'rgba(255, 255, 255, 0.8)' }}>Positive</span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ 
              width: '20px', 
              height: '20px', 
              backgroundColor: getSentimentColor(1),
              marginRight: '5px',
              borderRadius: '3px'
            }}></div>
            <span style={{ color: 'rgba(255, 255, 255, 0.8)' }}>Very Positive</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SentimentHeatmap;
