"""
LLM Oversight Demo for AI Trading Agent.

This script demonstrates the usage of the LLM oversight system
for autonomous trading decision validation and market analysis.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_trading_agent.oversight import (
    LLMOversight,
    OversightLevel,
    LLMProvider,
    OversightManager,
    TradingOversightAdapter,
    OrchestratorHookType,
    OversightConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables for the demo."""
    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OPENAI_API_KEY environment variable not set.")
        logger.warning("Please set your OpenAI API key to use the LLM oversight system.")
        logger.warning("You can obtain an API key from https://platform.openai.com/api-keys")
        
        # For demo purposes, we can ask for the key
        api_key = input("Enter your OpenAI API key (or press Enter to use a mock LLM): ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info("API key set for this session.")
            return True
        else:
            logger.info("No API key provided, using mock LLM responses.")
            return False
    
    logger.info("OPENAI_API_KEY environment variable is set.")
    return True


def create_mock_llm_response(prompt):
    """Create a mock LLM response for demo purposes when no API key is available."""
    # Extract key details from the prompt to create a reasonable mock response
    response = {
        "analysis": "This is a mock analysis generated without using an actual LLM API.",
        "decision": {},
        "explanation": "Mock explanation based on provided data."
    }
    
    # Basic pattern matching to determine the type of prompt
    if "market data" in prompt.lower():
        response["analysis"] = "Market analysis indicates moderate volatility with a slight bullish bias."
        response["market_regime"] = "moderately_bullish"
        response["explanation"] = "Based on price action and volume patterns, the market appears to be in a cautiously bullish regime."
    
    elif "trading decision" in prompt.lower() or "validate" in prompt.lower():
        if "buy" in prompt.lower():
            response["decision"] = {"action": "approve", "confidence": 0.75}
            response["explanation"] = "The buy decision aligns with the current market conditions and risk parameters."
        elif "sell" in prompt.lower():
            response["decision"] = {"action": "approve", "confidence": 0.82}
            response["explanation"] = "The sell decision is appropriate given the current portfolio exposure and market volatility."
        else:
            response["decision"] = {"action": "review", "confidence": 0.50}
            response["explanation"] = "Insufficient information to confidently approve this trading decision."
    
    elif "error" in prompt.lower() or "anomaly" in prompt.lower():
        response["analysis"] = "Detected potential issue in the data processing pipeline."
        response["recommendations"] = ["Verify data source connectivity", "Check for API rate limiting"]
        response["explanation"] = "The pattern of errors suggests an intermittent connection issue rather than a fundamental problem with the strategy logic."
    
    # Add timestamp for realism
    response["timestamp"] = datetime.now().isoformat()
    
    return response


class MockLLMOversight(LLMOversight):
    """Mock implementation of LLMOversight for demo without API key."""
    
    def generate_response(self, prompt, include_history=True):
        """Generate a mock response without calling an actual LLM API."""
        # Log the prompt for demo purposes
        logger.info(f"Mock LLM received prompt: {prompt[:100]}...")
        
        # Create a mock response
        mock_response = create_mock_llm_response(prompt)
        
        # Add to conversation history for realism
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({
            "role": "assistant", 
            "content": json.dumps(mock_response, indent=2)
        })
        
        return mock_response


def run_market_analysis_demo(use_real_llm=True):
    """Run a demonstration of market condition analysis."""
    logger.info("=== Market Analysis Demo ===")
    
    # Sample market data
    market_data = {
        "prices": {
            "SPY": [420.32, 422.67, 425.31, 423.89, 427.42],
            "QQQ": [370.12, 375.45, 380.22, 378.91, 384.56],
            "IWM": [198.76, 200.34, 201.56, 199.87, 202.43]
        },
        "volumes": {
            "SPY": [68245000, 72134000, 83245000, 65432000, 78654000],
            "QQQ": [45632000, 48765000, 52345000, 43567000, 50234000],
            "IWM": [32456000, 34567000, 38765000, 31245000, 36754000]
        },
        "indicators": {
            "vix": 18.45,
            "atr": {
                "SPY": 3.25,
                "QQQ": 4.12,
                "IWM": 2.87
            },
            "rsi": {
                "SPY": 58.34,
                "QQQ": 62.78,
                "IWM": 54.23
            },
            "moving_averages": {
                "SPY": {
                    "ma_20": 418.76,
                    "ma_50": 412.45,
                    "ma_200": 405.67
                },
                "QQQ": {
                    "ma_20": 368.43,
                    "ma_50": 362.78,
                    "ma_200": 354.32
                },
                "IWM": {
                    "ma_20": 197.45,
                    "ma_50": 195.67,
                    "ma_200": 190.23
                }
            }
        },
        "sentiment": {
            "twitter": 0.65,  # Positive sentiment score (0-1)
            "news": 0.58,
            "analyst_ratings": {
                "buy": 25,
                "hold": 15,
                "sell": 10
            }
        },
        "economic_indicators": {
            "interest_rates": 4.25,
            "unemployment": 3.9,
            "inflation_rate": 3.2,
            "gdp_growth": 2.1
        }
    }
    
    if use_real_llm:
        # Initialize the oversight system with real LLM
        oversight_manager = OversightManager(
            oversight_level=OversightLevel.ADVISE,
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-4"  # or "gpt-3.5-turbo" for a less expensive option
        )
    else:
        # Create a mock oversight system
        mock_llm = MockLLMOversight(
            provider=LLMProvider.CUSTOM,
            model_name="mock-gpt-4",
            oversight_level=OversightLevel.ADVISE
        )
        oversight_manager = OversightManager()
        oversight_manager.llm_oversight = mock_llm
    
    # Run market analysis
    logger.info("Analyzing market conditions...")
    result = oversight_manager.analyze_market_conditions(market_data)
    
    # Display results
    logger.info("\nMarket Analysis Results:")
    logger.info(f"Analysis: {result.get('analysis', 'No analysis provided')}")
    logger.info(f"Market Regime: {result.get('market_regime', 'Unknown')}")
    logger.info(f"Explanation: {result.get('explanation', 'No explanation provided')}")
    
    return result


def run_decision_validation_demo(use_real_llm=True):
    """Run a demonstration of trading decision validation."""
    logger.info("\n=== Trading Decision Validation Demo ===")
    
    # Sample trading decision
    decision = {
        "action": "BUY",
        "symbol": "AAPL",
        "quantity": 10,
        "price": 180.75,
        "order_type": "LIMIT",
        "time_in_force": "DAY",
        "strategy_name": "momentum_reversal",
        "strategy_parameters": {
            "entry_signal": "oversold_bounce",
            "confidence": 0.78,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.05
        },
        "risk_allocation": {
            "portfolio_pct": 0.03,
            "max_loss_usd": 450.00
        }
    }
    
    # Sample context
    context = {
        "market_conditions": {
            "market_regime": "neutral_with_positive_bias",
            "volatility": "moderate",
            "sector_performance": {
                "technology": 1.2,
                "healthcare": 0.8,
                "financials": -0.3
            }
        },
        "portfolio": {
            "cash": 25000.00,
            "equity": 75000.00,
            "positions": {
                "MSFT": {"quantity": 15, "avg_price": 340.50, "current_price": 345.20},
                "AMZN": {"quantity": 8, "avg_price": 155.75, "current_price": 162.30},
                "NVDA": {"quantity": 5, "avg_price": 450.25, "current_price": 492.75}
            },
            "sector_allocation": {
                "technology": 0.65,
                "healthcare": 0.15,
                "consumer_discretionary": 0.12,
                "other": 0.08
            }
        },
        "risk_metrics": {
            "portfolio_beta": 1.2,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.15,
            "value_at_risk_95": -0.03
        },
        "symbol_metrics": {
            "AAPL": {
                "beta": 1.05,
                "volatility_20d": 0.018,
                "rsi_14": 32.5,  # Oversold condition
                "price_to_ma_20": 0.92,  # Below 20-day moving average
                "analyst_consensus": "buy"
            }
        }
    }
    
    if use_real_llm:
        # Initialize the oversight system with real LLM
        oversight_manager = OversightManager(
            oversight_level=OversightLevel.APPROVE,
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-4"  # or "gpt-3.5-turbo" for a less expensive option
        )
    else:
        # Create a mock oversight system
        mock_llm = MockLLMOversight(
            provider=LLMProvider.CUSTOM,
            model_name="mock-gpt-4",
            oversight_level=OversightLevel.APPROVE
        )
        oversight_manager = OversightManager()
        oversight_manager.llm_oversight = mock_llm
    
    # Validate trading decision
    logger.info("Validating trading decision...")
    result = oversight_manager.validate_trading_decision(decision, context)
    
    # Display results
    logger.info("\nDecision Validation Results:")
    if isinstance(result.get('decision'), dict):
        action = result['decision'].get('action', 'unknown')
        logger.info(f"Decision: {action.upper()}")
        logger.info(f"Confidence: {result['decision'].get('confidence', 'N/A')}")
    else:
        logger.info(f"Decision: {result.get('decision', 'No decision provided')}")
    
    logger.info(f"Analysis: {result.get('analysis', 'No analysis provided')}")
    logger.info(f"Explanation: {result.get('explanation', 'No explanation provided')}")
    
    return result


def run_strategy_adjustment_demo(use_real_llm=True):
    """Run a demonstration of strategy adjustment suggestions."""
    logger.info("\n=== Strategy Adjustment Demo ===")
    
    # Sample current strategy configuration
    current_strategy = {
        "name": "adaptive_momentum",
        "description": "Adaptive momentum strategy with dynamic parameter adjustment",
        "base_parameters": {
            "lookback_period": 20,
            "momentum_threshold": 0.02,
            "volatility_adjustment": True,
            "entry_confirmation_signals": ["price_above_ma", "increasing_volume"],
            "position_sizing_method": "volatility_based"
        },
        "filters": {
            "min_volume": 500000,
            "max_spread_pct": 0.01,
            "exclude_earnings_days": True
        },
        "risk_parameters": {
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 4.0,
            "max_position_size_pct": 0.05,
            "max_sector_exposure_pct": 0.30,
            "correlation_threshold": 0.7
        },
        "adaptivity": {
            "regime_specific_settings": {
                "bullish": {
                    "momentum_threshold": 0.015,
                    "stop_loss_atr_multiple": 2.5,
                    "take_profit_atr_multiple": 5.0
                },
                "bearish": {
                    "momentum_threshold": 0.03,
                    "stop_loss_atr_multiple": 1.5,
                    "take_profit_atr_multiple": 3.0
                },
                "volatile": {
                    "momentum_threshold": 0.025,
                    "position_sizing_method": "reduced_volatility_based",
                    "stop_loss_atr_multiple": 1.5
                }
            }
        }
    }
    
    # Sample performance metrics
    performance_metrics = {
        "overall": {
            "total_return_pct": 12.5,
            "annualized_return_pct": 18.2,
            "sharpe_ratio": 1.35,
            "sortino_ratio": 1.85,
            "max_drawdown_pct": -8.5,
            "win_rate": 0.58,
            "profit_factor": 1.65,
            "avg_win_loss_ratio": 1.2,
            "volatility": 0.15
        },
        "by_regime": {
            "bullish": {
                "return_pct": 18.5,
                "win_rate": 0.72,
                "profit_factor": 2.1
            },
            "bearish": {
                "return_pct": -2.5,
                "win_rate": 0.35,
                "profit_factor": 0.85
            },
            "volatile": {
                "return_pct": 8.2,
                "win_rate": 0.51,
                "profit_factor": 1.35
            }
        },
        "by_sector": {
            "technology": {
                "return_pct": 21.5,
                "win_rate": 0.68
            },
            "healthcare": {
                "return_pct": 14.8,
                "win_rate": 0.62
            },
            "financials": {
                "return_pct": 6.5,
                "win_rate": 0.48
            },
            "energy": {
                "return_pct": -4.2,
                "win_rate": 0.40
            }
        },
        "recent_trades": [
            {"symbol": "MSFT", "return_pct": 3.2, "regime": "bullish", "holding_days": 12},
            {"symbol": "AMZN", "return_pct": 4.5, "regime": "bullish", "holding_days": 8},
            {"symbol": "JPM", "return_pct": -1.8, "regime": "volatile", "holding_days": 5},
            {"symbol": "XOM", "return_pct": -2.5, "regime": "bearish", "holding_days": 3},
            {"symbol": "NVDA", "return_pct": 7.2, "regime": "bullish", "holding_days": 14}
        ]
    }
    
    # Sample market conditions
    market_conditions = {
        "current_regime": "volatile",
        "regime_probabilities": {
            "bullish": 0.30,
            "bearish": 0.25,
            "volatile": 0.45
        },
        "regime_transition": {
            "previous": "bullish",
            "transition_date": "2024-05-10",
            "confidence": 0.75
        },
        "market_indicators": {
            "vix": 22.5,
            "average_sector_correlation": 0.68,
            "market_breadth": 0.55,
            "treasury_yield_10y": 3.8,
            "treasury_yield_change_1m": 0.3
        }
    }
    
    if use_real_llm:
        # Initialize the oversight system with real LLM
        oversight_manager = OversightManager(
            oversight_level=OversightLevel.ADVISE,
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
    else:
        # Create a mock oversight system
        mock_llm = MockLLMOversight(
            provider=LLMProvider.CUSTOM,
            model_name="mock-gpt-4",
            oversight_level=OversightLevel.ADVISE
        )
        oversight_manager = OversightManager()
        oversight_manager.llm_oversight = mock_llm
    
    # Get strategy adjustment suggestions
    logger.info("Generating strategy adjustment suggestions...")
    result = oversight_manager.suggest_strategy_adjustments(
        current_strategy, performance_metrics, market_conditions
    )
    
    # Display results
    logger.info("\nStrategy Adjustment Suggestions:")
    logger.info(f"Analysis: {result.get('analysis', 'No analysis provided')}")
    
    if 'suggestions' in result:
        logger.info("\nSpecific Suggestions:")
        for i, suggestion in enumerate(result['suggestions'], 1):
            logger.info(f"{i}. {suggestion.get('description', 'No description')}")
            if 'rationale' in suggestion:
                logger.info(f"   Rationale: {suggestion['rationale']}")
            if 'expected_impact' in suggestion:
                logger.info(f"   Expected Impact: {suggestion['expected_impact']}")
    
    logger.info(f"\nExplanation: {result.get('explanation', 'No explanation provided')}")
    
    return result


def main():
    """Run the LLM oversight demo."""
    logger.info("Starting LLM Oversight Demo")
    
    # Setup environment
    use_real_llm = setup_environment()
    
    # Run demos
    market_analysis = run_market_analysis_demo(use_real_llm)
    time.sleep(1)  # Small delay between API calls
    
    decision_validation = run_decision_validation_demo(use_real_llm)
    time.sleep(1)  # Small delay between API calls
    
    strategy_suggestions = run_strategy_adjustment_demo(use_real_llm)
    
    # Summary
    logger.info("\n=== Demo Summary ===")
    logger.info(f"Used real LLM: {use_real_llm}")
    logger.info("LLM oversight components demonstrated:")
    logger.info("1. Market regime analysis")
    logger.info("2. Trading decision validation")
    logger.info("3. Strategy adjustment suggestions")
    
    logger.info("\nDemo complete! The LLM oversight system is ready for integration.")


if __name__ == "__main__":
    main()
