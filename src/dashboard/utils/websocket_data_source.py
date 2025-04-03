"""
WebSocket Data Source

This module provides data sources for the WebSocket manager.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("websocket_data_source")

class WebSocketDataSource:
    """
    Base class for WebSocket data sources.
    """
    
    def __init__(self, data_service=None):
        """
        Initialize the WebSocket data source.
        
        Args:
            data_service: The data service to use for data retrieval
        """
        self.data_service = data_service
        logger.info(f"Initialized WebSocket data source: {self.__class__.__name__}")
    
    async def get_data(self) -> Dict[str, Any]:
        """
        Get data for the data source.
        
        Returns:
            The data
        """
        # This method should be overridden by subclasses
        return {}

class DashboardDataSource(WebSocketDataSource):
    """
    Data source for dashboard updates.
    """
    
    async def get_data(self) -> Dict[str, Any]:
        """
        Get dashboard data.
        
        Returns:
            The dashboard data
        """
        try:
            if self.data_service:
                # Get data from data service
                data = self.data_service.get_data('dashboard_summary')
                return data
            else:
                # Generate mock data
                return self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate mock dashboard data.
        
        Returns:
            The mock dashboard data
        """
        return {
            'total_value': f"${random.uniform(10000, 100000):.2f}",
            'daily_pnl': f"${random.uniform(-1000, 1000):.2f}",
            'open_positions': random.randint(1, 10),
            'win_rate': f"{random.uniform(40, 60):.1f}%",
            'sharpe_ratio': f"{random.uniform(0.5, 2.5):.2f}",
            'drawdown': f"{random.uniform(5, 15):.1f}%"
        }

class TradeDataSource(WebSocketDataSource):
    """
    Data source for trade updates.
    """
    
    def __init__(self, data_service=None):
        """
        Initialize the trade data source.
        
        Args:
            data_service: The data service to use for data retrieval
        """
        super().__init__(data_service)
        self.last_trade_id = 0
        self.symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'ADA/USD']
        self.sides = ['BUY', 'SELL']
    
    async def get_data(self) -> Dict[str, Any]:
        """
        Get trade data.
        
        Returns:
            The trade data
        """
        try:
            if self.data_service:
                # Get data from data service
                data = self.data_service.get_data('recent_trade')
                return data
            else:
                # Generate mock data
                return self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error getting trade data: {e}")
            return {}
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate mock trade data.
        
        Returns:
            The mock trade data
        """
        # Increment trade ID
        self.last_trade_id += 1
        
        # Generate random trade data
        symbol = random.choice(self.symbols)
        side = random.choice(self.sides)
        price = random.uniform(100, 50000)
        quantity = random.uniform(0.1, 10)
        value = price * quantity
        
        return {
            'id': self.last_trade_id,
            'time': datetime.now().strftime('%H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'price': f"${price:.2f}",
            'quantity': f"{quantity:.4f}",
            'value': f"${value:.2f}"
        }

class PositionDataSource(WebSocketDataSource):
    """
    Data source for position updates.
    """
    
    def __init__(self, data_service=None):
        """
        Initialize the position data source.
        
        Args:
            data_service: The data service to use for data retrieval
        """
        super().__init__(data_service)
        self.positions = {}
        self.symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'ADA/USD']
        self.sides = ['LONG', 'SHORT']
    
    async def get_data(self) -> Dict[str, Any]:
        """
        Get position data.
        
        Returns:
            The position data
        """
        try:
            if self.data_service:
                # Get data from data service
                data = self.data_service.get_data('position_update')
                return data
            else:
                # Generate mock data
                return self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error getting position data: {e}")
            return {}
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate mock position data.
        
        Returns:
            The mock position data
        """
        # Randomly decide whether to update an existing position or create a new one
        if self.positions and random.random() < 0.7:
            # Update existing position
            position_id = random.choice(list(self.positions.keys()))
            position = self.positions[position_id]
            
            # Update current price
            current_price = float(position['current_price'].replace('$', ''))
            price_change = current_price * random.uniform(-0.01, 0.01)
            new_price = current_price + price_change
            position['current_price'] = f"${new_price:.2f}"
            
            # Update PnL
            entry_price = float(position['entry_price'].replace('$', ''))
            quantity = float(position['quantity'])
            if position['side'] == 'LONG':
                pnl = (new_price - entry_price) * quantity
            else:
                pnl = (entry_price - new_price) * quantity
            position['pnl'] = f"${pnl:.2f}"
            
            return position
        else:
            # Create new position
            position_id = random.randint(1000, 9999)
            symbol = random.choice(self.symbols)
            side = random.choice(self.sides)
            quantity = random.uniform(0.1, 10)
            entry_price = random.uniform(100, 50000)
            current_price = entry_price * random.uniform(0.95, 1.05)
            
            if side == 'LONG':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            position = {
                'id': position_id,
                'symbol': symbol,
                'side': side,
                'quantity': f"{quantity:.4f}",
                'entry_price': f"${entry_price:.2f}",
                'current_price': f"${current_price:.2f}",
                'pnl': f"${pnl:.2f}"
            }
            
            # Store position
            self.positions[position_id] = position
            
            return position

class PerformanceDataSource(WebSocketDataSource):
    """
    Data source for performance updates.
    """
    
    async def get_data(self) -> Dict[str, Any]:
        """
        Get performance data.
        
        Returns:
            The performance data
        """
        try:
            if self.data_service:
                # Get data from data service
                data = self.data_service.get_data('performance_metrics')
                return data
            else:
                # Generate mock data
                return self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate mock performance data.
        
        Returns:
            The mock performance data
        """
        return {
            'daily_return': f"{random.uniform(-5, 5):.2f}%",
            'weekly_return': f"{random.uniform(-10, 10):.2f}%",
            'monthly_return': f"{random.uniform(-20, 20):.2f}%",
            'yearly_return': f"{random.uniform(-30, 30):.2f}%",
            'max_drawdown': f"{random.uniform(5, 15):.2f}%",
            'volatility': f"{random.uniform(10, 30):.2f}%",
            'sharpe_ratio': f"{random.uniform(0.5, 2.5):.2f}",
            'sortino_ratio': f"{random.uniform(0.5, 2.5):.2f}",
            'win_rate': f"{random.uniform(40, 60):.1f}%",
            'profit_factor': f"{random.uniform(1.0, 2.0):.2f}"
        }

class AlertDataSource(WebSocketDataSource):
    """
    Data source for alerts.
    """
    
    def __init__(self, data_service=None):
        """
        Initialize the alert data source.
        
        Args:
            data_service: The data service to use for data retrieval
        """
        super().__init__(data_service)
        self.alert_types = ['info', 'success', 'warning', 'error']
        self.alert_messages = [
            'New trade executed: {symbol} {side} at {price}',
            'Position closed: {symbol} with {pnl} profit',
            'Market volatility increased for {symbol}',
            'Stop loss triggered for {symbol} position',
            'Take profit reached for {symbol} position',
            'New market opportunity detected for {symbol}',
            'System performance degraded',
            'Connection to exchange restored',
            'Data validation failed for {symbol}',
            'Strategy {strategy} activated for {symbol}'
        ]
        self.symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'ADA/USD']
        self.strategies = ['Momentum', 'Mean Reversion', 'Breakout', 'Trend Following', 'Statistical Arbitrage']
        self.last_alert_time = datetime.now() - timedelta(minutes=5)
    
    async def get_data(self) -> Dict[str, Any]:
        """
        Get alert data.
        
        Returns:
            The alert data
        """
        try:
            if self.data_service:
                # Get data from data service
                data = self.data_service.get_data('system_alert')
                return data
            else:
                # Generate mock data
                # Only generate an alert occasionally
                if random.random() < 0.3:
                    return self._generate_mock_data()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error getting alert data: {e}")
            return {}
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate mock alert data.
        
        Returns:
            The mock alert data
        """
        # Generate random alert data
        alert_type = random.choice(self.alert_types)
        message_template = random.choice(self.alert_messages)
        
        # Format message with random data
        message = message_template.format(
            symbol=random.choice(self.symbols),
            side=random.choice(['BUY', 'SELL']),
            price=f"${random.uniform(100, 50000):.2f}",
            pnl=f"${random.uniform(-1000, 1000):.2f}",
            strategy=random.choice(self.strategies)
        )
        
        # Update last alert time
        self.last_alert_time = datetime.now()
        
        return {
            'type': alert_type,
            'message': message,
            'time': self.last_alert_time.strftime('%H:%M:%S')
        }