"""
Test script for the sentiment analysis system and its integration with technical analysis.

This script demonstrates how to:
1. Use the SentimentAnalysisAgent to analyze sentiment data
2. Use the IntegratedAnalysisAgent to combine sentiment and technical signals
3. Visualize the integrated trading signals
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SentimentIntegrationTest")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Visualization will be skipped.")
    MATPLOTLIB_AVAILABLE = False

# Import the necessary components
try:
    from ai_trading_agent.agent.sentiment_analysis_agent import SentimentAnalysisAgent, SentimentSource
    from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent, DataMode
    from ai_trading_agent.agent.integrated_analysis_agent import IntegratedAnalysisAgent, SignalWeight
    from ai_trading_agent.sentiment.data_sources import SentimentDataManager, DataSourceType
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)


def test_sentiment_analysis():
    """Test the standalone sentiment analysis agent."""
    logger.info("===== Testing Sentiment Analysis Agent =====")
    
    # Initialize the sentiment analysis agent with mock data
    symbols = ["BTC", "ETH"]
    sentiment_sources = [SentimentSource.MOCK]
    
    agent_config = {
        "logging": {"level": "INFO"},
        "topics": ["cryptocurrency", "blockchain", "defi"],
        "sentiment_analyzer": {
            "use_nltk": True,
            "use_textblob": True,
            "use_custom": True
        }
    }
    
    sentiment_agent = SentimentAnalysisAgent(
        agent_id_suffix="test",
        name="TestSentimentAgent",
        symbols=symbols,
        config_details=agent_config,
        sentiment_sources=sentiment_sources
    )
    
    # Process sentiment data
    signals = sentiment_agent.process()
    
    # Print results
    if signals:
        logger.info(f"Generated {len(signals)} sentiment signals")
        for signal in signals:
            symbol_or_topic = signal["payload"].get("symbol", signal["payload"].get("topic", "unknown"))
            strength = signal["payload"].get("signal_strength", 0)
            signal_type = signal["payload"].get("signal_type", "neutral")
            
            logger.info(f"Signal for {symbol_or_topic}: {signal_type.upper()} ({strength:.2f})")
    else:
        logger.warning("No sentiment signals generated")
    
    # Get metrics
    metrics = sentiment_agent.get_component_metrics()
    logger.info(f"Sentiment agent metrics: {metrics['agent']}")
    
    return sentiment_agent, signals


def test_integrated_analysis():
    """Test the integrated analysis agent that combines technical and sentiment signals."""
    logger.info("===== Testing Integrated Analysis Agent =====")
    
    # Initialize the integrated analysis agent
    symbols = ["BTC", "ETH"]
    
    agent_config = {
        "logging": {"level": "INFO"},
        "signal_weights": {
            "preset": "BALANCED"  # Equal weights for technical and sentiment
        },
        "technical_agent": {
            "data_mode": "mock"  # Use mock data for testing
        },
        "sentiment_agent": {
            "sentiment_sources": ["mock"]  # Use mock sentiment data
        }
    }
    
    integrated_agent = IntegratedAnalysisAgent(
        agent_id_suffix="test",
        name="TestIntegratedAgent",
        symbols=symbols,
        config_details=agent_config
    )
    
    # Process data to generate integrated signals
    signals = integrated_agent.process()
    
    # Print results
    if signals:
        logger.info(f"Generated {len(signals)} integrated signals")
        for signal in signals:
            symbol = signal["payload"].get("symbol", signal["payload"].get("topic", "unknown"))
            strength = signal["payload"].get("signal_strength", 0)
            signal_type = signal["payload"].get("signal_type", "neutral")
            tech_factor = signal["payload"].get("technical_factor", 0)
            sent_factor = signal["payload"].get("sentiment_factor", 0)
            
            logger.info(
                f"Integrated signal for {symbol}: {signal_type.upper()} ({strength:.2f})"
                f" [Tech: {tech_factor:.2f}, Sent: {sent_factor:.2f}]"
            )
    else:
        logger.warning("No integrated signals generated")
    
    # Get metrics
    metrics = integrated_agent.get_component_metrics()
    logger.info(f"Integrated agent metrics: {metrics['agent']}")
    
    return integrated_agent, signals


def test_different_weight_presets():
    """Test the integrated analysis agent with different weight presets."""
    logger.info("===== Testing Different Weight Presets =====")
    
    symbols = ["BTC", "ETH"]
    presets = ["TECHNICAL_ONLY", "SENTIMENT_ONLY", "BALANCED", 
               "TECHNICAL_PRIMARY", "SENTIMENT_PRIMARY"]
    
    results = {}
    
    for preset in presets:
        logger.info(f"Testing preset: {preset}")
        
        agent_config = {
            "logging": {"level": "INFO"},
            "signal_weights": {
                "preset": preset
            },
            "technical_agent": {
                "data_mode": "mock"
            },
            "sentiment_agent": {
                "sentiment_sources": ["mock"]
            }
        }
        
        integrated_agent = IntegratedAnalysisAgent(
            agent_id_suffix=f"test_{preset.lower()}",
            name=f"Test{preset}Agent",
            symbols=symbols,
            config_details=agent_config
        )
        
        signals = integrated_agent.process()
        
        if signals:
            signals_by_symbol = {}
            for signal in signals:
                if "payload" in signal and "symbol" in signal["payload"]:
                    symbol = signal["payload"]["symbol"]
                    signals_by_symbol[symbol] = signal["payload"]["signal_strength"]
            
            results[preset] = signals_by_symbol
            
            # Log the results
            logger.info(f"Results for {preset}:")
            for symbol, strength in signals_by_symbol.items():
                logger.info(f"  {symbol}: {strength:.2f}")
        else:
            logger.warning(f"No signals generated for {preset}")
    
    return results


def visualize_integrated_signals(results):
    """Visualize the integrated signals with different weight presets."""
    if not results:
        logger.warning("No results to visualize")
        return
        
    # Print the results in text format regardless of matplotlib availability
    logger.info("===== Integrated Signal Results =====")
    symbols = set()
    for preset_results in results.values():
        symbols.update(preset_results.keys())
    
    symbols = sorted(list(symbols))
    presets = list(results.keys())
    
    # Print a text-based table
    header = "Symbol   " + "   ".join(f"{preset:15s}" for preset in presets)
    logger.info(header)
    logger.info("-" * len(header))
    
    for symbol in symbols:
        row = f"{symbol:8s}"
        for preset in presets:
            value = results[preset].get(symbol, 0)
            row += f"   {value:+.2f}        "
        logger.info(row)
    
    # If matplotlib is available, create a visualization
    if MATPLOTLIB_AVAILABLE:
        try:
            # Set up the plot
            plt.figure(figsize=(12, 8))
            bar_width = 0.15
            opacity = 0.8
            index = np.arange(len(symbols))
            
            # Plot bars for each preset
            for i, preset in enumerate(presets):
                values = [results[preset].get(symbol, 0) for symbol in symbols]
                plt.bar(
                    index + i * bar_width, 
                    values,
                    bar_width,
                    alpha=opacity,
                    label=preset
                )
            
            plt.xlabel('Symbols')
            plt.ylabel('Signal Strength')
            plt.title('Integrated Signal Strength by Weight Preset')
            plt.xticks(index + bar_width * (len(presets) - 1) / 2, symbols)
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('integrated_signals.png')
            logger.info("Saved visualization to integrated_signals.png")
            
            # Try to display the plot
            try:
                plt.show()
            except Exception as e:
                logger.warning(f"Could not display plot: {e}")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            traceback.print_exc()
    else:
        logger.warning("Matplotlib not available, skipping visualization")
        # Save results to CSV as an alternative
        try:
            df = pd.DataFrame(index=symbols)
            for preset in presets:
                df[preset] = [results[preset].get(symbol, 0) for symbol in symbols]
            df.to_csv('integrated_signals.csv')
            logger.info("Saved results to integrated_signals.csv")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")


def main():
    """Run all tests."""
    logger.info("Starting sentiment integration tests")
    
    try:
        # Test sentiment analysis
        logger.info("Running sentiment analysis test")
        sentiment_agent, sentiment_signals = test_sentiment_analysis()
        logger.info("Sentiment analysis test completed")
        
        # Test integrated analysis
        logger.info("Running integrated analysis test")
        integrated_agent, integrated_signals = test_integrated_analysis()
        logger.info("Integrated analysis test completed")
        
        # Test different weight presets
        logger.info("Running weight preset tests")
        preset_results = test_different_weight_presets()
        logger.info("Weight preset tests completed")
        
        # Visualize the results
        logger.info("Visualizing results")
        visualize_integrated_signals(preset_results)
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        logger.info("=== SENTIMENT ANALYSIS INTEGRATION TEST ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current directory: {os.getcwd()}")
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        traceback.print_exc()
        sys.exit(1)
