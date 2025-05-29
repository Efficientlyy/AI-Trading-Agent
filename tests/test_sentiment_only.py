import os
import sys
import unittest

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_agent.agent.sentiment_analysis_agent import SentimentAnalysisAgent
from ai_trading_agent.agent.agent_definitions import AgentRole, AgentStatus

class TestSentimentAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and agent instances"""
        # Define test symbols
        self.symbols = ['BTC', 'ETH']
        
        # Create sentiment agent instance
        self.sentiment_agent = SentimentAnalysisAgent(
            agent_id_suffix="test",
            name="Test Sentiment Agent",
            symbols=self.symbols,
            config_details={
                "sentiment_sources": ["news", "social"],
                "sentiment_window": 7  # days
            }
        )
    
    def test_sentiment_agent_initialization(self):
        """Test if the sentiment agent initializes correctly"""
        # Verify the agent was created with the correct parameters
        self.assertEqual(self.sentiment_agent.name, "Test Sentiment Agent")
        self.assertEqual(self.sentiment_agent.agent_type, "SentimentAnalysis")
        self.assertEqual(self.sentiment_agent.symbols, self.symbols)
        
        # Verify agent configuration
        self.assertIn("sentiment_sources", self.sentiment_agent.config_details)
        self.assertIn("sentiment_window", self.sentiment_agent.config_details)
        
        # Verify agent role and status
        self.assertEqual(self.sentiment_agent.agent_role, AgentRole.SPECIALIZED_TECHNICAL)
        self.assertEqual(self.sentiment_agent.status, AgentStatus.INITIALIZING)
        
        print(f"\nSuccessfully initialized sentiment agent: {self.sentiment_agent.name}")
    
    def test_sentiment_agent_generates_signals(self):
        """Test if the sentiment agent can generate signals"""
        # Generate signals (mock data will be used internally)
        signals = self.sentiment_agent.generate_signals()
        
        # Verify signals were generated
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, list)
        
        print(f"\nGenerated {len(signals)} sentiment signals")
        
        # Print sample signals
        if signals:
            for i, signal in enumerate(signals[:3]):  # Print first 3 signals
                print(f"Signal {i+1}: {signal}")
        
        # Verify agent state after signal generation
        self.assertEqual(self.sentiment_agent.status, AgentStatus.READY)

if __name__ == '__main__':
    unittest.main()
