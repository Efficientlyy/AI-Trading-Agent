import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading_agent.agent.factory import (
    create_data_manager,
    create_strategy,
    create_risk_manager,
    create_orchestrator,
    create_strategy_manager
)
from ai_trading_agent.agent.strategy import SimpleStrategyManager
from ai_trading_agent.agent.portfolio import SimplePortfolioManager
from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler

# Dummy config compatibility checker
def check_config_compatibility(config):
    return []

logging.basicConfig(
    filename="minimal_backtest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
logger.propagate = True
logger.info("TEST LOG OUTPUT: Script started")

def generate_sample_data(symbols, start_date, end_date, data_dir):
    logger.info("Generating dummy price data and saving to disk...")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    for symbol in symbols:
        df = pd.DataFrame(index=dates)
        df['timestamp'] = df.index.strftime('%Y-%m-%d')
        df['open'] = np.random.uniform(100, 200, size=len(dates))
        df['high'] = df['open'] + np.random.uniform(0, 10, size=len(dates))
        df['low'] = df['open'] - np.random.uniform(0, 10, size=len(dates))
        df['close'] = df['open'] + np.random.uniform(-5, 5, size=len(dates))
        df['volume'] = np.random.randint(1000, 10000, size=len(dates))
        df = df.reset_index(drop=True)
        filename = os.path.join(data_dir, f"{symbol}_ohlcv_1d.csv")
        df.to_csv(filename, index=False)
        logger.info(f"Saved OHLCV data for {symbol} to {filename}")
    return True

def generate_sample_sentiment(symbols, start_date, end_date, data_dir):
    logger.info("Generating dummy sentiment data and saving to disk...")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    sentiment_rows = []
    for symbol in symbols:
        sentiments = np.random.uniform(-1, 1, size=len(dates))
        for dt, sentiment in zip(dates, sentiments):
            sentiment_rows.append({
                'timestamp': dt.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'sentiment': sentiment
            })
    df = pd.DataFrame(sentiment_rows)
    sentiment_file = os.path.join(data_dir, 'synthetic_sentiment.csv')
    df.to_csv(sentiment_file, index=False)
    logger.info(f"Saved sentiment data to {sentiment_file}")
    return True

def run_backtest_with_factory():
    logger.info("Starting minimal backtest with factory system")
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    symbols = ["AAPL", "GOOG", "MSFT"]
    initial_capital = 100000.0
    data_dir = os.path.join('temp_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory: {data_dir}")
    else:
        logger.info(f"Data directory already exists: {data_dir}")
    generate_sample_data(symbols, start_date, end_date, data_dir)
    generate_sample_sentiment(symbols, start_date, end_date, data_dir)
    logger.info("Sample data generated and saved.")
    agent_config = {
        "data_manager": {
            "type": "SimpleDataManager",
            "config": {
                "data_dir": data_dir,
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": "1d",
                "data_types": ["ohlcv", "sentiment"]
            }
        },
        "strategy": {
            "type": "SentimentStrategy",
            "config": {
                "name": "SentimentStrategy",
                "symbols": symbols,
                "buy_threshold": 0.1,
                "sell_threshold": -0.1,
                "position_size_pct": 0.1
            }
        },
        "risk_manager": {
            "type": "SimpleRiskManager",
            "config": {},
        },
        "backtest": {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "signal_processing": {
                "sentiment_filter": "ema",
                "sentiment_filter_window": 10,
                "price_filter": "savgol",
                "price_filter_window": 15,
                "regime_detection": "volatility",
                "regime_window": 20,
                "regime_vol_threshold": 0.025
            },
            "available_strategies": [
                'SentimentStrategy',
                'MovingAverageCrossover',
                'RSIStrategy',
                'BollingerBandsStrategy',
                'ConservativeStrategy',
                'AggressiveStrategy',
                'MultiStrategyEnsemble',
                'OptimizedMAC'
            ],
            "optimizer": {
                "type": "GAOptimizer",
                "config": {
                    "fitness_fn": "dummy_fitness",
                    "param_space": {
                        'sentiment_threshold': [0.1, 0.2, 0.3, 0.4],
                        'position_size_pct': [0.05, 0.1, 0.15, 0.2]
                    }
                }
            }
        }
    }
    logger.info("Agent config created.")
    warnings = check_config_compatibility(agent_config)

    # --- Merge advanced config into strategy config ---
    for k in ["signal_processing", "symbols", "available_strategies", "optimizer"]:
        if k in agent_config["backtest"]:
            agent_config["strategy"]["config"][k] = agent_config["backtest"][k]
    # --- Ensure essential fields are present ---
    for k in ["symbols", "start_date", "end_date"]:
        if k in agent_config["data_manager"]["config"]:
            agent_config["strategy"]["config"][k] = agent_config["data_manager"]["config"][k]

    for warning in warnings:
        logger.warning(warning)
    data_manager = create_data_manager(agent_config["data_manager"])
    logger.info("Data manager created.")
    strategy = create_strategy(agent_config["strategy"])
    logger.info(f"Strategy created: {strategy}")
    # Ensure all required fields are present in manager_config (build directly from agent_config)
    manager_config = dict(agent_config["strategy"]["config"])
    for k in ["symbols", "start_date", "end_date"]:
        if k in agent_config["data_manager"]["config"]:
            manager_config[k] = agent_config["data_manager"]["config"][k]
    strategy_manager = create_strategy_manager(strategy, manager_type="SentimentStrategyManager", data_manager=data_manager, config=manager_config)
    logger.info("Strategy manager created via factory and configured.")
    portfolio_manager = SimplePortfolioManager(initial_cash=initial_capital, config={})
    logger.info("Portfolio manager created.")
    risk_manager = create_risk_manager(agent_config["risk_manager"])
    logger.info("Risk manager created.")
    execution_handler = SimulatedExecutionHandler(portfolio_manager=portfolio_manager, config={})
    logger.info("Execution handler created.")
    from ai_trading_agent.optimization.ga_optimizer import GAOptimizer
    def dummy_fitness(params):
        return 1.0
    param_space = {
        'sentiment_threshold': [0.1, 0.2, 0.3, 0.4],
        'position_size_pct': [0.05, 0.1, 0.15, 0.2]
    }
    optimizer = GAOptimizer(fitness_fn=dummy_fitness, param_space=param_space)
    available_strategies = [
        'SentimentStrategy',
        'MovingAverageCrossover',
        'RSIStrategy',
        'BollingerBandsStrategy',
        'ConservativeStrategy',
        'AggressiveStrategy',
        'MultiStrategyEnsemble',
        'OptimizedMAC'
    ]
    # --- Advanced signal processing config example ---
    agent_config["backtest"]["signal_processing"] = {
        "sentiment_filter": "ema",
        "sentiment_filter_window": 10,
        "price_filter": "savgol",
        "price_filter_window": 15,
        "regime_detection": "volatility",
        "regime_window": 20,
        "regime_vol_threshold": 0.025
    }
    agent_config["backtest"]["available_strategies"] = available_strategies
    agent_config["backtest"]["optimizer"] = optimizer
    logger.info("Adaptive intelligence configured.")
    orchestrator = create_orchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=agent_config["backtest"]
    )
    logger.info("Orchestrator created.")
    logger.info("Running backtest...")
    results = orchestrator.run()
    logger.info("Backtest finished.")
    if results:
        logger.info(f"Backtest completed successfully")
        print("\n=== Backtest Performance Metrics ===")
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            logger.info(f"Performance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                    print(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
                    print(f"  {key}: {value}")
        if 'adaptive_reason' in results:
            logger.info(f"Adaptive Reason: {results['adaptive_reason']}")
            print(f"Adaptive Reason: {results['adaptive_reason']}")
        print("====================================\n")
    else:
        logger.warning("Backtest did not return results")
        print("Backtest did not return results.")
    return results

if __name__ == "__main__":
    run_backtest_with_factory()
