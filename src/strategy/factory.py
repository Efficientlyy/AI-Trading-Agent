"""Strategy factory for creating trading strategies.

This module provides a factory for creating and registering trading
strategies in the AI Crypto Trading Agent platform.
"""

from typing import Dict, Type, Optional, Any

from src.common.config import config
from src.common.logging import get_logger
from src.strategy.base_strategy import Strategy


class StrategyFactory:
    """Factory for creating trading strategies.
    
    This class provides a centralized way to create strategy instances
    based on configuration.
    """
    
    # Registry of available strategies
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[Strategy]) -> None:
        """Register a new strategy class.
        
        Args:
            name: The name of the strategy
            strategy_class: The strategy class to register
        """
        cls._strategies[name] = strategy_class
        get_logger("strategy", "factory").info(f"Registered strategy: {name}")
    
    @classmethod
    def create(cls, strategy_type: str, strategy_id: Optional[str] = None, **kwargs: Any) -> Optional[Strategy]:
        """Create a strategy instance.
        
        Args:
            strategy_type: The type of strategy to create
            strategy_id: Optional custom ID for the strategy
            **kwargs: Additional arguments to pass to the strategy constructor
            
        Returns:
            The created strategy instance, or None if the strategy type is not found
        """
        logger = get_logger("strategy", "factory")
        
        if strategy_type not in cls._strategies:
            logger.error("Unknown strategy type", strategy_type=strategy_type)
            return None
            
        strategy_class = cls._strategies[strategy_type]
        
        try:
            if strategy_id:
                strategy = strategy_class(strategy_id=strategy_id, **kwargs)
            else:
                strategy = strategy_class(**kwargs)
                
            logger.info("Created strategy", 
                      type=strategy_type, 
                      id=strategy.strategy_id)
                
            return strategy
            
        except Exception as e:
            logger.error("Failed to create strategy", 
                       type=strategy_type,
                       error=str(e))
            return None
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, Type[Strategy]]:
        """Get all available strategy types.
        
        Returns:
            Dictionary of strategy names to strategy classes
        """
        return cls._strategies.copy()


# Import strategies to register them
from src.strategy.sentiment_strategy import SentimentStrategy
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy

# Register default strategies
StrategyFactory.register("sentiment", SentimentStrategy)
StrategyFactory.register("enhanced_sentiment", EnhancedSentimentStrategy)

# Import additional strategies if they exist
try:
    from src.strategy.ma_crossover import MACrossoverStrategy
    StrategyFactory.register("ma_crossover", MACrossoverStrategy)
except ImportError:
    pass

try:
    from src.strategy.market_imbalance import MarketImbalanceStrategy
    StrategyFactory.register("market_imbalance", MarketImbalanceStrategy)
except ImportError:
    pass

try:
    from src.strategy.meta_strategy import MetaStrategy
    StrategyFactory.register("meta", MetaStrategy)
except ImportError:
    pass