import React, { useEffect, useState } from 'react';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { useNotification } from '../../../components/common/NotificationSystem';
import webSocketService, { WebSocketTopic } from '../../../services/WebSocketService';
import { format } from 'date-fns';

interface Trade {
  symbol: string;
  quantity: number;
  price: number;
  timestamp: string;
  side: 'buy' | 'sell';
  id?: string;
}

const PaperTradingTradeNotifications: React.FC = () => {
  const { state } = usePaperTrading();
  const { showNotification } = useNotification();
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    if (state.activeSessions.length > 0) {
      // Connect to WebSocket
      webSocketService.connect([WebSocketTopic.TRADES, WebSocketTopic.STATUS])
        .then(() => {
          setIsConnected(true);
          console.log('Connected to WebSocket for trade notifications');
        })
        .catch(error => {
          console.error('Failed to connect to WebSocket for trade notifications:', error);
        });

      // Set up event handlers
      const handleTradeUpdate = (data: { trades: Trade[] }) => {
        if (!data.trades || !Array.isArray(data.trades)) return;

        // Add unique IDs to trades if they don't have them
        const tradesWithIds = data.trades.map(trade => ({
          ...trade,
          id: trade.id || `${trade.symbol}-${trade.timestamp}-${Math.random().toString(36).substring(2, 9)}`
        }));

        // Update recent trades
        setRecentTrades(prev => {
          // Merge new trades with existing trades, avoiding duplicates
          const existingIds = new Set(prev.map(t => t.id));
          const newTrades = tradesWithIds.filter(t => !existingIds.has(t.id));
          
          // Combine and sort by timestamp (newest first)
          const combined = [...prev, ...newTrades].sort((a, b) => 
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );
          
          // Keep only the most recent 50 trades
          return combined.slice(0, 50);
        });

        // Show notifications for new trades
        tradesWithIds.forEach(trade => {
          const action = trade.side === 'buy' ? 'Bought' : 'Sold';
          const message = `${action} ${Math.abs(trade.quantity)} ${trade.symbol} @ ${trade.price}`;
          
          showNotification({
            title: 'Trade Executed',
            message,
            type: 'info',
            duration: 5000
          });
        });
      };

      const handleStatusUpdate = (data: any) => {
        if (data.recent_trades && Array.isArray(data.recent_trades)) {
          // Add unique IDs to trades if they don't have them
          const tradesWithIds = data.recent_trades.map((trade: Trade) => ({
            ...trade,
            id: trade.id || `${trade.symbol}-${trade.timestamp}-${Math.random().toString(36).substring(2, 9)}`
          }));

          // Set recent trades
          setRecentTrades(tradesWithIds);
        }
      };

      // Register event handlers
      webSocketService.on(WebSocketTopic.TRADES, handleTradeUpdate);
      webSocketService.on(WebSocketTopic.STATUS, handleStatusUpdate);

      // Cleanup function
      return () => {
        webSocketService.off(WebSocketTopic.TRADES, handleTradeUpdate);
        webSocketService.off(WebSocketTopic.STATUS, handleStatusUpdate);
        
        // Disconnect from WebSocket
        webSocketService.disconnect();
        setIsConnected(false);
      };
    }
  }, [state.activeSessions.length, showNotification]);

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format date
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return format(date, 'yyyy-MM-dd HH:mm:ss');
    } catch (error) {
      return dateString;
    }
  };

  return (
    <div className="paper-trading-trade-notifications">
      <div className="notifications-header">
        <h3>Recent Trades</h3>
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      
      <div className="trades-list">
        {recentTrades.length === 0 ? (
          <div className="no-data">No recent trades</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Quantity</th>
                <th>Price</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {recentTrades.map(trade => (
                <tr key={trade.id} className={trade.side}>
                  <td>{formatDate(trade.timestamp)}</td>
                  <td>{trade.symbol}</td>
                  <td className={trade.side}>{trade.side.toUpperCase()}</td>
                  <td>{Math.abs(trade.quantity)}</td>
                  <td>{formatCurrency(trade.price)}</td>
                  <td>{formatCurrency(Math.abs(trade.quantity * trade.price))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default PaperTradingTradeNotifications;
