#!/usr/bin/env python
"""
End-to-End Testing Script for LLM Oversight Integration.

This script simulates a production environment and tests the integration
between the trading orchestrator and the LLM oversight service.
"""

import os
import sys
import time
import json
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, List, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from ai_trading_agent.oversight.client import OversightClient, OversightAction
from ai_trading_agent.agent.adaptive_orchestrator import AdaptiveHealthOrchestrator
from ai_trading_agent.market_regime import MarketRegimeType, VolatilityRegimeType
from ai_trading_agent.market_regime.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/oversight_integration_test.log")
    ]
)
logger = logging.getLogger("oversight_integration_test")


def generate_sample_market_data() -> Dict[str, pd.DataFrame]:
    """Generate sample market data for testing."""
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']
    data = {}
    
    # Generate data for multiple assets
    for symbol in symbols:
        # Generate some random price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), end=datetime.now(), freq='D')
        n = len(dates)
        
        # Base price varies by symbol
        base_price = {
            'AAPL': 150,
            'MSFT': 300,
            'AMZN': 3000,
            'GOOG': 2500,
            'TSLA': 800
        }.get(symbol, 100)
        
        # Create different price patterns
        if symbol in ['AAPL', 'MSFT']:
            # Uptrend
            close = [base_price * (1 + 0.001 * i + 0.005 * (i % 5 - 2)) for i in range(n)]
        elif symbol == 'AMZN':
            # Downtrend
            close = [base_price * (1 - 0.0005 * i + 0.006 * (i % 6 - 3)) for i in range(n)]
        elif symbol == 'GOOG':
            # Sideways
            close = [base_price * (1 + 0.001 * (i % 10 - 5)) for i in range(n)]
        else:
            # Volatile
            close = [base_price * (1 + 0.002 * i + 0.02 * (i % 3 - 1)) for i in range(n)]
        
        # Create Open, High, Low prices
        open_price = [c * (1 + 0.002 * (i % 2 - 0.5)) for i, c in enumerate(close)]
        high = [max(o, c) * (1 + 0.01) for o, c in zip(open_price, close)]
        low = [min(o, c) * (1 - 0.01) for o, c in zip(open_price, close)]
        
        # Generate volume
        volume = [1000000 * (1 + 0.2 * (i % 5)) for i in range(n)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change().fillna(0)
        
        data[symbol] = df
        
    return data


def generate_sample_trading_decisions(symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Generate sample trading decisions for testing."""
    strategies = ['trend', 'mean_reversion', 'momentum', 'ml_strategy']
    
    decisions = {}
    for strategy in strategies:
        strategy_decisions = []
        
        for symbol in symbols:
            # Generate different actions based on strategy
            if strategy == 'trend':
                action = 'buy' if symbol in ['AAPL', 'MSFT', 'GOOG'] else 'sell'
                confidence = 0.8 if symbol in ['AAPL', 'MSFT'] else 0.6
            elif strategy == 'mean_reversion':
                action = 'sell' if symbol in ['AAPL', 'MSFT', 'GOOG'] else 'buy'
                confidence = 0.7 if symbol in ['AMZN', 'TSLA'] else 0.5
            elif strategy == 'momentum':
                action = 'buy' if symbol in ['AAPL', 'MSFT', 'TSLA'] else 'hold'
                confidence = 0.85 if symbol == 'TSLA' else 0.7
            else:  # ml_strategy
                action = 'buy' if symbol in ['GOOG', 'MSFT'] else 'sell'
                confidence = 0.9 if symbol == 'GOOG' else 0.75
            
            # Skip 'hold' actions
            if action == 'hold':
                continue
                
            # Generate sample prices based on action
            base_price = {
                'AAPL': 150,
                'MSFT': 300,
                'AMZN': 3000,
                'GOOG': 2500,
                'TSLA': 800
            }.get(symbol, 100)
            
            # Add some variation to prices
            price = base_price * (1 + 0.01 * (hash(f"{strategy}_{symbol}") % 10 - 5) / 100)
            
            # Generate decision
            decision = {
                'symbol': symbol,
                'action': action,
                'quantity': 10 if action == 'buy' else 5,
                'price': price,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'strategy': strategy,
                'timeframe': '1d',
                'metadata': {
                    'signal_type': 'price_action' if strategy in ['trend', 'momentum'] else 'oscillator',
                    'parameters': {
                        'lookback_periods': 20,
                        'threshold': 0.5
                    }
                }
            }
            
            strategy_decisions.append(decision)
        
        decisions[strategy] = strategy_decisions
    
    return decisions


def test_oversight_health(client: OversightClient) -> bool:
    """Test oversight service health."""
    logger.info("Testing oversight service health...")
    try:
        is_healthy = client.check_health()
        if is_healthy:
            logger.info("✅ Oversight service is healthy")
        else:
            logger.error("❌ Oversight service is unhealthy")
        return is_healthy
    except Exception as e:
        logger.error(f"❌ Error checking oversight service health: {e}")
        return False


def test_market_analysis(client: OversightClient, market_data: Dict[str, pd.DataFrame]) -> bool:
    """Test market analysis functionality."""
    logger.info("Testing market analysis...")
    try:
        # Convert market data to format expected by oversight service
        analysis_data = {}
        for symbol, df in market_data.items():
            # Convert last 30 days of data
            recent_data = df.tail(30).copy()
            analysis_data[symbol] = {
                'date': recent_data.index.strftime('%Y-%m-%d').tolist(),
                'open': recent_data['open'].tolist(),
                'high': recent_data['high'].tolist(),
                'low': recent_data['low'].tolist(),
                'close': recent_data['close'].tolist(),
                'volume': recent_data['volume'].tolist(),
                'returns': recent_data['returns'].tolist()
            }
        
        # Send to oversight service
        analysis_result = client.analyze_market_conditions(analysis_data)
        
        # Check result
        if analysis_result and 'regime' in analysis_result:
            logger.info(f"✅ Market analysis successful - Detected regime: {analysis_result['regime']}")
            logger.info(f"   Confidence: {analysis_result.get('confidence', 'N/A')}")
            
            # Log key insights if available
            if 'key_insights' in analysis_result:
                logger.info("   Key insights:")
                for insight in analysis_result['key_insights']:
                    logger.info(f"   - {insight}")
            
            return True
        else:
            logger.error("❌ Market analysis failed - Invalid response format")
            return False
    except Exception as e:
        logger.error(f"❌ Error during market analysis: {e}")
        return False


def test_decision_validation(
    client: OversightClient, 
    decisions: Dict[str, List[Dict[str, Any]]],
    market_context: Dict[str, Any]
) -> bool:
    """Test decision validation functionality."""
    logger.info("Testing decision validation...")
    success = True
    
    try:
        # Test validation for each strategy
        for strategy, strategy_decisions in decisions.items():
            logger.info(f"Testing decisions for strategy: {strategy}")
            
            for decision in strategy_decisions:
                # Add validation context
                context = market_context.copy()
                context['strategy'] = strategy
                
                # Get validation action
                try:
                    action = client.get_decision_action(decision, context)
                    validation_result = client.validate_trading_decision(decision, context)
                    
                    # Log results
                    symbol = decision['symbol']
                    trade_action = decision['action']
                    logger.info(f"  {symbol} {trade_action.upper()}: {action.name} - {validation_result.get('reason', 'No reason provided')}")
                    
                    # Check if we have detailed recommendations
                    if 'recommendation' in validation_result:
                        logger.info(f"    Recommendation: {validation_result['recommendation']}")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error validating decision for {decision['symbol']}: {e}")
                    success = False
        
        return success
    except Exception as e:
        logger.error(f"❌ Error during decision validation tests: {e}")
        return False


def test_orchestrator_integration(
    oversight_url: str,
    market_data: Dict[str, pd.DataFrame],
    trading_decisions: Dict[str, List[Dict[str, Any]]]
) -> bool:
    """Test integration with adaptive orchestrator."""
    logger.info("Testing integration with adaptive orchestrator...")
    
    try:
        # Create orchestrator with LLM oversight enabled
        orchestrator = AdaptiveHealthOrchestrator(
            enable_llm_oversight=True,
            llm_oversight_service_url=oversight_url,
            llm_oversight_level="advise"  # Start with advise level
        )
        
        # Set market data in orchestrator
        # Convert DataFrame to dict format expected by orchestrator
        simplified_market_data = {}
        for symbol, df in market_data.items():
            simplified_market_data[symbol] = {
                'close': df['close'].values.tolist(),
                'volume': df['volume'].values.tolist(),
                'returns': df['returns'].values.tolist()
            }
        orchestrator.market_data = simplified_market_data
        
        # Set portfolio state
        orchestrator.portfolio_value = 100000
        orchestrator.portfolio_drawdown = 0.02
        orchestrator.current_regime = {
            "global": {
                "regime_type": MarketRegimeType.BULLISH.value,
                "volatility_type": VolatilityRegimeType.NORMAL.value,
                "confidence": 0.85,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Test validation functionality
        logger.info("Testing decision validation via orchestrator...")
        for strategy, decisions in trading_decisions.items():
            for decision in decisions:
                is_approved, result = orchestrator.validate_trading_decision(decision)
                symbol = decision['symbol']
                action = decision['action']
                logger.info(f"  {symbol} {action.upper()}: {'APPROVED' if is_approved else 'REJECTED'} - {result.get('reason', 'No reason')}")
        
        # Test cycle with oversight
        logger.info("Testing orchestrator cycle with LLM oversight...")
        
        # Mock the parent run_cycle method to isolate testing
        original_run_cycle = orchestrator.run_cycle
        orchestrator.run_cycle = lambda inputs=None: {"status": "success", "decisions_processed": len(inputs) if inputs else 0}
        
        try:
            # Run cycle with trading decisions
            cycle_result = orchestrator.run_cycle(trading_decisions)
            logger.info(f"  Cycle completed: {cycle_result}")
            
            # Test with different oversight levels
            for level in ["monitor", "approve", "autonomous"]:
                logger.info(f"Testing with oversight level: {level}")
                orchestrator.llm_oversight_level = level
                cycle_result = orchestrator.run_cycle(trading_decisions)
                logger.info(f"  Cycle with {level} level completed")
            
            return True
        finally:
            # Restore original method
            orchestrator.run_cycle = original_run_cycle
        
    except Exception as e:
        logger.error(f"❌ Error during orchestrator integration test: {e}")
        return False


def main():
    """Run the end-to-end tests."""
    parser = argparse.ArgumentParser(description='Test LLM Oversight Integration')
    parser.add_argument('--url', type=str, default='http://localhost:8080',
                        help='URL of the oversight service')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("STARTING LLM OVERSIGHT INTEGRATION TESTS")
    logger.info("="*80)
    
    # Create output directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize client
    client = OversightClient(base_url=args.url)
    
    # Generate test data
    logger.info("Generating test data...")
    market_data = generate_sample_market_data()
    symbols = list(market_data.keys())
    trading_decisions = generate_sample_trading_decisions(symbols)
    
    # Create market context for validation
    market_context = {
        "market_regime": {
            "global": {
                "regime_type": "BULLISH",
                "volatility_type": "NORMAL",
                "confidence": 0.85
            }
        },
        "portfolio": {
            "value": 100000,
            "drawdown": 0.02,
            "allocation": {
                "AAPL": 0.15,
                "MSFT": 0.20,
                "AMZN": 0.10,
                "GOOG": 0.15,
                "TSLA": 0.05
            }
        },
        "risk_metrics": {
            "portfolio_risk": 0.02,
            "max_position_size": 0.25,
            "in_stress_mode": False
        }
    }
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_oversight_health(client)),
        ("Market Analysis", lambda: test_market_analysis(client, market_data)),
        ("Decision Validation", lambda: test_decision_validation(client, trading_decisions, market_context)),
        ("Orchestrator Integration", lambda: test_orchestrator_integration(args.url, market_data, trading_decisions))
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("\n" + "-"*50)
        logger.info(f"Running test: {test_name}")
        logger.info("-"*50)
        
        start_time = time.time()
        try:
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
            logger.info(f"{test_name}: {'✅ PASSED' if success else '❌ FAILED'} in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            logger.error(f"{test_name}: ❌ ERROR - {str(e)} in {duration:.2f}s")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    for test_name, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name.ljust(25)}: {status} ({duration:.2f}s)")
    
    # Final result
    all_passed = all(success for _, success, _ in results)
    logger.info("\nOverall Result: " + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
