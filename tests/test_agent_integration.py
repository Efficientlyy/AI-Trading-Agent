import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent
from ai_trading_agent.agent.sentiment_analysis_agent import SentimentAnalysisAgent
from ai_trading_agent.agent.integrated_analysis_agent import IntegratedAnalysisAgent
from ai_trading_agent.agent.agent_definitions import AgentRole, AgentStatus

class TestAgentIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and agent instances"""
        # Create sample price data for testing
        self.dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                    end=datetime.now(), 
                                    freq='D')
        self.prices = np.random.normal(100, 5, len(self.dates))
        self.volumes = np.random.randint(1000, 10000, len(self.dates))
        
        # Create a DataFrame with sample price data
        self.price_data = pd.DataFrame({
            'timestamp': self.dates,
            'open': self.prices,
            'high': self.prices + np.random.normal(0, 1, len(self.dates)),
            'low': self.prices - np.random.normal(0, 1, len(self.dates)),
            'close': self.prices + np.random.normal(0, 0.5, len(self.dates)),
            'volume': self.volumes
        })
        
        # Set timestamp as index
        self.price_data.set_index('timestamp', inplace=True)
        
        # Define test symbols
        self.symbols = ['BTC', 'ETH']
        
        # Create agent instances
        self.technical_agent = TechnicalAnalysisAgent(
            agent_id_suffix="test",
            name="Test Technical Agent",
            agent_type="Technical",
            symbols=self.symbols,
            config_details={
                "indicators": {
                    "RSI": {"period": 14, "enabled": True},
                    "EMA": {"period": 20, "enabled": True},
                    "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "enabled": True}
                }
            }
        )
        
        self.sentiment_agent = SentimentAnalysisAgent(
            agent_id_suffix="test",
            name="Test Sentiment Agent",
            agent_type="Sentiment",
            symbols=self.symbols,
            config_details={
                "sentiment_sources": ["news", "social"],
                "sentiment_window": 7  # days
            }
        )
        
        self.integrated_agent = IntegratedAnalysisAgent(
            agent_id_suffix="test",
            name="Test Integrated Agent",
            symbols=self.symbols,
            config_details={
                "technical_weight": 0.6,
                "sentiment_weight": 0.4,
                "signal_threshold": 0.5
            }
        )
    
    def test_technical_agent_generates_signals(self):
        """Test if the technical agent can generate signals from price data"""
        # Set agent data
        self.technical_agent.set_data(self.price_data)
        
        # Generate signals
        signals = self.technical_agent.generate_signals()
        
        # Verify signals were generated
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, list)
        
        print(f"\nGenerated {len(signals)} technical signals")
        
        # Verify agent state after signal generation
        self.assertEqual(self.technical_agent.status, AgentStatus.READY)
    
    def test_sentiment_agent_generates_signals(self):
        """Test if the sentiment agent can generate signals from sentiment data"""
        # Generate signals (mock data will be used internally)
        signals = self.sentiment_agent.generate_signals()
        
        # Verify signals were generated
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, list)
        
        print(f"\nGenerated {len(signals)} sentiment signals")
        
        # Verify agent state after signal generation
        self.assertEqual(self.sentiment_agent.status, AgentStatus.READY)
    
    def test_integrated_agent_combines_signals(self):
        """Test if the integrated agent can combine signals from technical and sentiment agents"""
        # Set up technical agent
        self.technical_agent.set_data(self.price_data)
        technical_signals = self.technical_agent.generate_signals()
        
        # Get sentiment signals
        sentiment_signals = self.sentiment_agent.generate_signals()
        
        # Set child agents for integrated agent
        self.integrated_agent.technical_agent = self.technical_agent
        self.integrated_agent.sentiment_agent = self.sentiment_agent
        
        # Generate combined signals
        combined_signals = self.integrated_agent.generate_signals()
        
        # Verify combined signals were generated
        self.assertIsNotNone(combined_signals)
        self.assertIsInstance(combined_signals, list)
        
        print(f"\nGenerated {len(combined_signals)} combined signals")
        
        # Verify agent state after signal generation
        self.assertEqual(self.integrated_agent.status, AgentStatus.READY)

if __name__ == '__main__':
    unittest.main()
