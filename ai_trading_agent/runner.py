"""
AI Trading Agent Runner

Main entry point for the AI Trading Agent system with Market Regime Classification
and Adaptive Response capabilities fully integrated.
"""

import os
import logging
import argparse
import yaml
from typing import Dict, Any, Optional

from ai_trading_agent.agent.adaptive_orchestrator import AdaptiveHealthOrchestrator
from ai_trading_agent.agent.data_agent import MarketDataAgent
from ai_trading_agent.agent.signal_generator import SignalGeneratorAgent
from ai_trading_agent.agent.portfolio_manager import PortfolioManagerAgent
from ai_trading_agent.agent.risk_manager import RiskManagerAgent
from ai_trading_agent.agent.execution_agent import ExecutionAgent
from ai_trading_agent.agent.meta_strategy import DynamicAggregationMetaStrategy
from ai_trading_agent.agent.adaptive_manager import AdaptiveStrategyManager
from ai_trading_agent.market_regime import MarketRegimeConfig

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AITradingAgentRunner:
    """
    Main runner class for the AI Trading Agent system.
    
    This class initializes and runs the complete AI Trading Agent system with
    Market Regime Classification and Adaptive Response capabilities fully integrated.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        adaptation_interval: int = 60
    ):
        """
        Initialize the AI Trading Agent Runner.
        
        Args:
            config_path: Path to configuration file
            log_dir: Directory for logs
            adaptation_interval: Interval for market regime adaptation (minutes)
        """
        self.config = self._load_config(config_path)
        self.log_dir = log_dir or self.config.get('log_dir', './logs')
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create the adaptive health orchestrator
        regime_config = self._create_regime_config()
        self.orchestrator = AdaptiveHealthOrchestrator(
            log_dir=self.log_dir,
            regime_config=regime_config,
            temporal_pattern_enabled=self.config.get('enable_temporal_patterns', True),
            adaptation_interval_minutes=adaptation_interval
        )
        
        # Initialize the agents
        self._initialize_agents()
        
        logger.info("AI Trading Agent Runner initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
            return {}
    
    def _create_regime_config(self) -> MarketRegimeConfig:
        """Create market regime configuration from config file."""
        regime_config = MarketRegimeConfig()
        
        # Apply configuration from file if available
        if 'market_regime' in self.config:
            rc = self.config['market_regime']
            
            # Configure volatility thresholds
            if 'volatility_thresholds' in rc:
                vt = rc['volatility_thresholds']
                regime_config.volatility_low_threshold = vt.get('low', 0.01)
                regime_config.volatility_medium_threshold = vt.get('medium', 0.02)
                regime_config.volatility_high_threshold = vt.get('high', 0.03)
            
            # Configure momentum thresholds
            if 'momentum_thresholds' in rc:
                mt = rc['momentum_thresholds']
                regime_config.strong_momentum_threshold = mt.get('strong', 0.1)
                regime_config.weak_momentum_threshold = mt.get('weak', 0.05)
            
            # Configure other parameters
            regime_config.lookback_window = rc.get('lookback_window', 20)
            regime_config.enable_volatility_clustering = rc.get('enable_volatility_clustering', True)
            regime_config.enable_correlation_regimes = rc.get('enable_correlation_regimes', True)
            regime_config.enable_liquidity_analysis = rc.get('enable_liquidity_analysis', True)
            
        return regime_config
    
    def _initialize_agents(self):
        """Initialize and register all agents with the orchestrator."""
        # Initialize the market data agent
        data_agent = MarketDataAgent(
            tickers=self.config.get('tickers', ['SPY', 'QQQ', 'IWM']),
            data_sources=self.config.get('data_sources', ['yfinance']),
            update_interval=self.config.get('data_update_interval', 60)
        )
        self.orchestrator.register_agent('market_data', data_agent)
        self.orchestrator.register_market_data_source('market_data', data_agent)
        
        # Initialize the signal generator agent
        signal_generator = SignalGeneratorAgent(
            indicators=self.config.get('indicators', ['rsi', 'macd', 'bollinger']),
            timeframes=self.config.get('timeframes', ['1d', '4h', '1h'])
        )
        self.orchestrator.register_agent('signal_generator', signal_generator)
        
        # Initialize the adaptive strategy manager
        adaptive_manager = AdaptiveStrategyManager(
            strategy_pool=self.config.get('strategies', ['momentum', 'mean_reversion', 'trend_following']),
            default_strategy=self.config.get('default_strategy', 'trend_following')
        )
        self.orchestrator.register_agent('adaptive_manager', adaptive_manager)
        
        # Initialize the meta-strategy agent
        meta_strategy = DynamicAggregationMetaStrategy(
            methods=self.config.get('aggregation_methods', ['majority_vote', 'weighted_average', 'dynamic_threshold'])
        )
        self.orchestrator.register_agent('meta_strategy', meta_strategy)
        
        # Initialize the risk manager agent
        risk_manager = RiskManagerAgent(
            max_portfolio_risk=self.config.get('max_portfolio_risk', 0.02),
            max_position_size=self.config.get('max_position_size', 0.1),
            risk_free_rate=self.config.get('risk_free_rate', 0.01)
        )
        self.orchestrator.register_agent('risk_manager', risk_manager)
        
        # Initialize the portfolio manager agent
        portfolio_manager = PortfolioManagerAgent(
            initial_capital=self.config.get('initial_capital', 100000),
            risk_manager=risk_manager
        )
        self.orchestrator.register_agent('portfolio_manager', portfolio_manager)
        
        # Initialize the execution agent
        execution_agent = ExecutionAgent(
            broker=self.config.get('broker', 'paper'),
            api_key=self.config.get('api_key', ''),
            api_secret=self.config.get('api_secret', '')
        )
        self.orchestrator.register_agent('execution', execution_agent)
        
        # Set dependencies between agents
        self.orchestrator.set_dependency('signal_generator', 'market_data')
        self.orchestrator.set_dependency('adaptive_manager', 'signal_generator')
        self.orchestrator.set_dependency('meta_strategy', 'adaptive_manager')
        self.orchestrator.set_dependency('risk_manager', 'meta_strategy')
        self.orchestrator.set_dependency('portfolio_manager', 'risk_manager')
        self.orchestrator.set_dependency('execution', 'portfolio_manager')
        
        logger.info(f"Initialized {len(self.orchestrator.agents)} agents")
    
    def start(self):
        """Start the AI Trading Agent system."""
        logger.info("Starting AI Trading Agent system...")
        
        try:
            # Start all agents
            self.orchestrator.start_all_agents()
            
            # Start the main orchestration loop
            self.orchestrator.start()
            
            logger.info("AI Trading Agent system started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start AI Trading Agent system: {str(e)}")
            return False
    
    def stop(self):
        """Stop the AI Trading Agent system."""
        logger.info("Stopping AI Trading Agent system...")
        
        try:
            # Stop the orchestration
            self.orchestrator.stop()
            
            # Stop all agents
            self.orchestrator.stop_all_agents()
            
            logger.info("AI Trading Agent system stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Error stopping AI Trading Agent system: {str(e)}")
            return False


def main():
    """Main entry point for the AI Trading Agent system."""
    parser = argparse.ArgumentParser(description="AI Trading Agent System")
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--log-dir', type=str, help="Directory for logs")
    parser.add_argument('--adaptation-interval', type=int, default=60,
                      help="Interval for market regime adaptation (minutes)")
    
    args = parser.parse_args()
    
    runner = AITradingAgentRunner(
        config_path=args.config,
        log_dir=args.log_dir,
        adaptation_interval=args.adaptation_interval
    )
    
    runner.start()
    
    # Keep the main thread running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runner.stop()


if __name__ == "__main__":
    main()
