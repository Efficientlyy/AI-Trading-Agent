import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from ai_trading_agent.agent.agent_definitions import (
    BaseAgent, AgentRole, AgentStatus,
    SentimentAnalysisAgent, TechnicalAnalysisAgent, NewsEventAgent,
    FundamentalAnalysisAgent, DecisionAgent
)

class TestBaseAgent(unittest.TestCase):

    def test_base_agent_initialization_defaults(self):
        agent = BaseAgent(
            agent_id="test_base_001",
            name="Test Base Agent",
            agent_role=AgentRole.DECISION_AGGREGATOR,
            agent_type="BaseTestType"
        )
        self.assertEqual(agent.agent_id, "test_base_001")
        self.assertEqual(agent.name, "Test Base Agent")
        self.assertEqual(agent.agent_role, AgentRole.DECISION_AGGREGATOR)
        self.assertEqual(agent.agent_type, "BaseTestType")
        self.assertEqual(agent.status, AgentStatus.IDLE)
        self.assertEqual(agent.inputs_from, [])
        self.assertEqual(agent.outputs_to, [])
        self.assertEqual(agent.config_details, {})
        self.assertEqual(agent.metrics, {})
        self.assertIsInstance(agent.last_updated, datetime)
        self.assertEqual(agent.symbols, [])

    def test_base_agent_initialization_with_values(self):
        config = {"param1": "value1"}
        metrics_init = {"runs": 0}
        inputs = ["source_agent_1"]
        outputs = ["target_agent_1"]
        symbols_monitored = ["BTC/USD"]
        
        agent = BaseAgent(
            agent_id="test_base_002",
            name="Configured Base Agent",
            agent_role=AgentRole.SPECIALIZED_SENTIMENT,
            agent_type="ConfigTestType",
            status=AgentStatus.INITIALIZING,
            inputs_from=inputs,
            outputs_to=outputs,
            config_details=config,
            metrics=metrics_init,
            symbols=symbols_monitored
        )
        self.assertEqual(agent.status, AgentStatus.INITIALIZING)
        self.assertEqual(agent.inputs_from, inputs)
        self.assertEqual(agent.outputs_to, outputs)
        self.assertEqual(agent.config_details, config)
        self.assertEqual(agent.metrics, metrics_init)
        self.assertEqual(agent.symbols, symbols_monitored)

    def test_update_status(self):
        agent = BaseAgent("test_status_001", "Status Agent", AgentRole.EXECUTION_BROKER, "StatusTester")
        old_timestamp = agent.last_updated
        agent.update_status(AgentStatus.RUNNING)
        self.assertEqual(agent.status, AgentStatus.RUNNING)
        self.assertGreater(agent.last_updated, old_timestamp)

    def test_update_metrics(self):
        agent = BaseAgent("test_metrics_001", "Metrics Agent", AgentRole.SPECIALIZED_TECHNICAL, "MetricsTester")
        agent.update_metrics({"new_metric": 100})
        self.assertEqual(agent.metrics.get("new_metric"), 100)
        agent.update_metrics({"new_metric": 150, "another_metric": 50})
        self.assertEqual(agent.metrics.get("new_metric"), 150)
        self.assertEqual(agent.metrics.get("another_metric"), 50)

    def test_get_info(self):
        agent = BaseAgent(
            agent_id="test_info_001",
            name="Info Agent",
            agent_role=AgentRole.SPECIALIZED_NEWS,
            agent_type="InfoTester"
        )
        agent.update_status(AgentStatus.RUNNING)
        info = agent.get_info()
        
        self.assertEqual(info["agent_id"], "test_info_001")
        self.assertEqual(info["name"], "Info Agent")
        self.assertEqual(info["agent_role"], AgentRole.SPECIALIZED_NEWS.value)
        self.assertEqual(info["type"], "InfoTester")
        self.assertEqual(info["status"], AgentStatus.RUNNING.value)
        self.assertIn("last_updated", info)

    def test_process_method_not_implemented(self):
        agent = BaseAgent("test_process_001", "Process Agent", AgentRole.SPECIALIZED_FUNDAMENTAL, "ProcessTester")
        with self.assertRaises(NotImplementedError):
            agent.process()

    def test_start_stop_methods(self):
        agent = BaseAgent("test_start_stop_001", "Lifecycle Agent", AgentRole.DECISION_AGGREGATOR, "LifecycleTester")
        self.assertEqual(agent.status, AgentStatus.IDLE)
        agent.start()
        self.assertEqual(agent.status, AgentStatus.RUNNING)
        agent.stop()
        self.assertEqual(agent.status, AgentStatus.STOPPED)

# Example of a test for a derived class (can be expanded)
class TestSentimentAnalysisAgent(unittest.TestCase):
    def test_sentiment_agent_initialization(self):
        agent = SentimentAnalysisAgent(
            agent_id_suffix="test_sentiment",
            name="Test Sentiment",
            agent_type="TestSentimentType",
            symbols=["SYM/BOL"],
            config_details={"api_key": "test_key"}
        )
        self.assertEqual(agent.agent_id, "spec_sentiment_test_sentiment")
        self.assertEqual(agent.agent_role, AgentRole.SPECIALIZED_SENTIMENT)
        self.assertEqual(agent.symbols, ["SYM/BOL"])
        self.assertEqual(agent.api_key, "test_key")

    @patch('ai_trading_agent.agent.agent_definitions.SentimentAnalysisAgent._fetch_news_data')
    @patch('ai_trading_agent.agent.agent_definitions.SentimentAnalysisAgent._analyze_sentiment_from_news')
    def test_sentiment_agent_process_no_external_data(self, mock_analyze, mock_fetch):
        mock_fetch.return_value = [{"title": "Test News"}] # Simulate fetched data
        mock_analyze.return_value = {"symbol": "SYM/BOL", "sentiment_score": 0.5, "trend": "bullish"} # Simulate analysis
        
        agent = SentimentAnalysisAgent(
            agent_id_suffix="process_test", name="SA Process Test",
            agent_type="TestProcessor", symbols=["SYM/BOL"]
        )
        agent.start() # To set status to RUNNING, though process doesn't check status internally
        
        result = agent.process()
        
        mock_fetch.assert_called_once_with("SYM/BOL")
        mock_analyze.assert_called_once_with([{"title": "Test News"}], "SYM/BOL")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "sentiment_signal")
        self.assertEqual(result[0]["payload"]["sentiment_score"], 0.5)
        self.assertIn("total_signals_generated", agent.metrics)

    @patch('ai_trading_agent.agent.agent_definitions.SentimentAnalysisAgent._fetch_news_data')
    def test_sentiment_agent_process_no_news_fetched(self, mock_fetch):
        mock_fetch.return_value = None # Simulate no news found
        agent = SentimentAnalysisAgent(
            agent_id_suffix="no_news", name="SA No News",
            agent_type="TestNoNews", symbols=["SYM/BOL"]
        )
        result = agent.process()
        self.assertIsNone(result)
        mock_fetch.assert_called_once_with("SYM/BOL")

    def test_sentiment_agent_process_with_external_data(self):
        agent = SentimentAnalysisAgent(
            agent_id_suffix="ext_data", name="SA Ext Data",
            agent_type="TestExtData", symbols=["SYM/BOL"]
        )
        external_signal = {"type": "pre_analyzed_sentiment", "payload": {"symbol": "EXT/SYM", "score": 0.8}}
        result = agent.process(external_signal)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["payload"]["score"], 0.8)


class TestTechnicalAnalysisAgent(unittest.TestCase):
    def test_technical_agent_initialization(self):
        agent = TechnicalAnalysisAgent(
            agent_id_suffix="test_tech", name="Test Tech",
            agent_type="TestTechType", symbols=["TECH/USD"],
            config_details={"rsi_period": 21}
        )
        self.assertEqual(agent.agent_id, "spec_technical_test_tech")
        self.assertEqual(agent.agent_role, AgentRole.SPECIALIZED_TECHNICAL)
        self.assertEqual(agent.symbols, ["TECH/USD"])
        self.assertEqual(agent.rsi_period, 21)

    @patch('ai_trading_agent.agent.agent_definitions.TechnicalAnalysisAgent._fetch_market_data')
    @patch('ai_trading_agent.agent.agent_definitions.TechnicalAnalysisAgent._calculate_indicators')
    def test_technical_agent_process_no_external_data(self, mock_calculate, mock_fetch):
        mock_fetch.return_value = {"close": 100} # Simulate fetched market data
        mock_calculate.return_value = {"symbol": "TECH/USD", "indicator_rsi": 60, "signal": "buy"} # Simulate calculation
        
        agent = TechnicalAnalysisAgent(
            agent_id_suffix="process_test_tech", name="TA Process Test",
            agent_type="TestTechProcessor", symbols=["TECH/USD"]
        )
        result = agent.process()
        
        mock_fetch.assert_called_once_with("TECH/USD")
        mock_calculate.assert_called_once_with({"close": 100}, "TECH/USD")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "technical_signal")
        self.assertEqual(result[0]["payload"]["signal"], "buy")
        self.assertIn("total_indicators_calculated", agent.metrics)

    @patch('ai_trading_agent.agent.agent_definitions.TechnicalAnalysisAgent._fetch_market_data')
    def test_technical_agent_process_no_market_data_fetched(self, mock_fetch):
        mock_fetch.return_value = None # Simulate no market data found
        agent = TechnicalAnalysisAgent(
            agent_id_suffix="no_mkt_data", name="TA No Market Data",
            agent_type="TestNoMarketData", symbols=["TECH/USD"]
        )
        result = agent.process()
        self.assertIsNone(result)
        mock_fetch.assert_called_once_with("TECH/USD")

    def test_technical_agent_process_with_external_data(self):
        agent = TechnicalAnalysisAgent(
            agent_id_suffix="ext_data_tech", name="TA Ext Data",
            agent_type="TestExtDataTech", symbols=["TECH/USD"]
        )
        # Mock the _calculate_indicators since it's called by process when data is provided
        agent._calculate_indicators = MagicMock(return_value={"symbol": "EXT/TECH", "signal": "sell"})
        
        external_market_data = {"symbol": "EXT/TECH", "close": 200} # External data must include symbol
        result = agent.process(external_market_data)
        
        agent._calculate_indicators.assert_called_once_with(external_market_data, "EXT/TECH")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["payload"]["signal"], "sell")


class TestNewsEventAgent(unittest.TestCase):
    def test_news_event_agent_initialization(self):
        agent = NewsEventAgent(
            agent_id_suffix="test_news", name="Test NewsEvent",
            agent_type="TestNewsType", symbols=["STOCKA"],
            event_keywords=["earnings"], config_details={"news_api_key": "key123"}
        )
        self.assertEqual(agent.agent_id, "spec_news_test_news")
        self.assertEqual(agent.agent_role, AgentRole.SPECIALIZED_NEWS)
        self.assertEqual(agent.symbols, ["STOCKA"])
        self.assertIn("earnings", agent.event_keywords)
        self.assertEqual(agent.news_api_key, "key123")

    @patch('ai_trading_agent.agent.agent_definitions.NewsEventAgent._fetch_news_events')
    @patch('ai_trading_agent.agent.agent_definitions.NewsEventAgent._analyze_event_impact')
    def test_news_event_agent_process_fetches_and_analyzes(self, mock_analyze, mock_fetch):
        mock_fetch.return_value = [{"title": "Event 1", "symbols_affected": ["STOCKA"]}]
        mock_analyze.return_value = {"event_title": "Event 1", "potential_impact_score": 0.7}
        
        agent = NewsEventAgent(agent_id_suffix="news_proc", name="News Processor", agent_type="TypeN", symbols=["STOCKA"])
        result = agent.process()

        mock_fetch.assert_called_once_with("STOCKA")
        mock_analyze.assert_called_once_with({"title": "Event 1", "symbols_affected": ["STOCKA"]})
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["type"], "news_event_signal")
        self.assertEqual(result[0]["payload"]["potential_impact_score"], 0.7)
        self.assertIn("total_events_processed", agent.metrics)


class TestFundamentalAnalysisAgent(unittest.TestCase):
    def test_fundamental_agent_initialization(self):
        agent = FundamentalAnalysisAgent(
            agent_id_suffix="test_funda", name="Test Funda",
            agent_type="TestFundaType", symbols=["STOCKB"],
            config_details={"financial_data_api_key": "f_key"}
        )
        self.assertEqual(agent.agent_id, "spec_fundamental_test_funda")
        self.assertEqual(agent.agent_role, AgentRole.SPECIALIZED_FUNDAMENTAL)
        self.assertEqual(agent.symbols, ["STOCKB"])
        self.assertEqual(agent.financial_data_provider_key, "f_key")

    @patch('ai_trading_agent.agent.agent_definitions.FundamentalAnalysisAgent._fetch_fundamental_data')
    @patch('ai_trading_agent.agent.agent_definitions.FundamentalAnalysisAgent._analyze_fundamentals')
    def test_fundamental_agent_process_fetches_and_analyzes(self, mock_analyze, mock_fetch):
        mock_fetch.return_value = {"pe_ratio": 10}
        mock_analyze.return_value = {"symbol": "STOCKB", "valuation_score": 0.8}

        agent = FundamentalAnalysisAgent(agent_id_suffix="funda_proc", name="Funda Processor", agent_type="TypeF", symbols=["STOCKB"])
        result = agent.process()

        mock_fetch.assert_called_once_with("STOCKB")
        mock_analyze.assert_called_once_with({"pe_ratio": 10}, "STOCKB")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["type"], "fundamental_signal")
        self.assertEqual(result[0]["payload"]["valuation_score"], 0.8)
        self.assertIn("total_analyses_completed", agent.metrics)


class TestDecisionAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            "min_signals_for_decision": 2,
            "buy_threshold": 0.6,
            "sell_threshold": -0.6,
            "signal_weights": {
                "sentiment_signal": 0.5,
                "technical_signal": 0.5
            },
            "risk_management": {
                "default_trade_quantity": 1.0,
                "max_trade_value_usd": 1000.0,
                "per_symbol_max_quantity": {"TEST/USD": 10}
            }
        }
        self.agent = DecisionAgent(
            agent_id_suffix="test_decision", name="Test Decision Agent",
            agent_type="TestDecisionType", config_details=self.config
        )

    def test_decision_agent_initialization(self):
        self.assertEqual(self.agent.signal_weights["sentiment_signal"], 0.5)
        self.assertEqual(self.agent.buy_threshold, 0.6)
        self.assertEqual(self.agent.risk_config["max_trade_value_usd"], 1000.0)

    def test_decision_agent_process_buy_decision(self):
        sentiment_signal = {"type": "sentiment_signal", "payload": {"symbol": "TEST/USD", "sentiment_score": 0.8}}
        technical_signal = {"type": "technical_signal", "payload": {"symbol": "TEST/USD", "signal": "buy", "price_at_signal": 90.0}} # price for risk check

        # Process first signal
        self.agent.process(sentiment_signal)
        # Process second signal, should trigger decision
        result = self.agent.process(technical_signal)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "trading_directive")
        self.assertEqual(result["payload"]["action"], "buy")
        self.assertEqual(result["payload"]["symbol"], "TEST/USD")
        self.assertAlmostEqual(result["payload"]["quantity"], 1.0) # Default quantity, within risk limits

    def test_decision_agent_process_sell_decision(self):
        sentiment_signal = {"type": "sentiment_signal", "payload": {"symbol": "TEST/USD", "sentiment_score": -0.7}}
        technical_signal = {"type": "technical_signal", "payload": {"symbol": "TEST/USD", "signal": "sell", "price_at_signal": 110.0}}
        
        self.agent.process(sentiment_signal)
        result = self.agent.process(technical_signal)

        self.assertIsNotNone(result)
        self.assertEqual(result["payload"]["action"], "sell")

    def test_decision_agent_process_hold_decision_score_neutral(self):
        sentiment_signal = {"type": "sentiment_signal", "payload": {"symbol": "TEST/USD", "sentiment_score": 0.1}}
        technical_signal = {"type": "technical_signal", "payload": {"symbol": "TEST/USD", "signal": "neutral"}}
        
        self.agent.process(sentiment_signal)
        result = self.agent.process(technical_signal)
        self.assertIsNone(result) # Hold decision results in None

    def test_decision_agent_process_hold_decision_not_enough_signals(self):
        sentiment_signal = {"type": "sentiment_signal", "payload": {"symbol": "TEST/USD", "sentiment_score": 0.9}}
        result = self.agent.process(sentiment_signal)
        self.assertIsNone(result) # Not enough unique signal types yet

    def test_decision_agent_risk_max_trade_value(self):
        # Configure for a smaller max value to trigger adjustment
        self.agent.max_trade_value_usd = 50.0
        self.agent.default_trade_quantity = 1.0

        sentiment_signal = {"type": "sentiment_signal", "payload": {"symbol": "TEST/USD", "sentiment_score": 0.8}}
        # Price is 90, 1.0 * 90 = 90, which is > 50. Quantity should be 50/90 = 0.555...
        technical_signal = {"type": "technical_signal", "payload": {"symbol": "TEST/USD", "signal": "buy", "price_at_signal": 90.0}}
        
        self.agent.process(sentiment_signal)
        result = self.agent.process(technical_signal)

        self.assertIsNotNone(result)
        self.assertEqual(result["payload"]["action"], "buy")
        self.assertAlmostEqual(result["payload"]["quantity"], 50.0/90.0, places=7)

    def test_decision_agent_risk_per_symbol_max_quantity(self):
        self.agent.per_symbol_max_quantity = {"TEST/USD": 0.5}
        self.agent.default_trade_quantity = 1.0 # Default is higher than symbol max

        sentiment_signal = {"type": "sentiment_signal", "payload": {"symbol": "TEST/USD", "sentiment_score": 0.8}}
        technical_signal = {"type": "technical_signal", "payload": {"symbol": "TEST/USD", "signal": "buy", "price_at_signal": 90.0}}
        
        self.agent.process(sentiment_signal)
        result = self.agent.process(technical_signal)

        self.assertIsNotNone(result)
        self.assertEqual(result["payload"]["action"], "buy")
        self.assertEqual(result["payload"]["quantity"], 0.5)


if __name__ == '__main__':
    unittest.main()