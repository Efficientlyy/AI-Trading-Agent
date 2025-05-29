"""
Alternative Data Integration Example

This script demonstrates how to use the Alternative Data Integration module
to fetch and analyze data from satellite imagery, social media, and supply chain sources.
"""
import asyncio
import json
from datetime import datetime, timedelta
import os
from pprint import pprint

from ai_trading_agent.data_sources.alternative_data import (
    AlternativeDataConfig,
    AlternativeDataIntegration,
    SatelliteImageryAnalyzer,
    SocialMediaSentimentAnalyzer,
    SupplyChainDataAnalyzer
)


async def main():
    print("Alternative Data Integration Example")
    print("====================================")
    
    # Create configurations for each data source
    # In a real scenario, these would be loaded from environment variables or a config file
    satellite_config = AlternativeDataConfig(
        api_key="satellite_api_key_example",
        endpoint="https://api.planet.com/data/v1/"
    )
    
    social_media_config = AlternativeDataConfig(
        api_key="social_media_api_key_example",
        endpoint="https://api.twitter.com/2/"
    )
    
    supply_chain_config = AlternativeDataConfig(
        api_key="supply_chain_api_key_example",
        endpoint="https://api.freightos.com/v1/"
    )
    
    # Initialize the integration module with all data sources
    alt_data = AlternativeDataIntegration(
        satellite_config=satellite_config,
        social_media_config=social_media_config,
        supply_chain_config=supply_chain_config
    )
    
    # Define query parameters for each data source
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    
    query_params = {
        "satellite": {
            "analysis_type": "oil_storage",
            "location": {
                "lat": 29.7604,
                "lon": -95.3698,
                "region": "houston_tx"
            },
            "start_date": one_month_ago.isoformat(),
            "end_date": today.isoformat()
        },
        "social_media": {
            "symbols": ["AAPL", "TSLA", "NVDA"],
            "keywords": ["inflation", "recession", "fed"],
            "platforms": ["twitter", "reddit"],
            "start_date": one_month_ago.isoformat(),
            "end_date": today.isoformat()
        },
        "supply_chain": {
            "data_type": "freight_rates",
            "routes": ["china_to_us_west", "china_to_europe"],
            "start_date": one_month_ago.isoformat(),
            "end_date": today.isoformat()
        }
    }
    
    print("\nFetching data from all sources...")
    data_dict = await alt_data.fetch_all_data(query_params)
    
    # Show summary of retrieved data
    print("\nData summary:")
    for source, data in data_dict.items():
        if not data.empty:
            print(f"- {source}: {len(data)} records")
            print(f"  Columns: {', '.join(data.columns)}")
        else:
            print(f"- {source}: No data retrieved")
    
    # Process signals from all sources
    print("\nProcessing signals...")
    signals = alt_data.process_signals(data_dict)
    
    # Display aggregated signal
    print("\nAggregated Signal:")
    print(f"Direction: {signals['signal']}")
    print(f"Strength: {signals['strength']:.2f}")
    print(f"Timestamp: {signals['timestamp']}")
    
    # Display weights
    print("\nSource Weights:")
    for source, weight in signals.get('source_weights', {}).items():
        print(f"- {source}: {weight:.2f}")
    
    # Display source signals
    print("\nSource Signals:")
    for source, signal in signals.get('source_signals', {}).items():
        print(f"- {source}: {signal['signal']} (strength: {signal['strength']:.2f})")
    
    # Display top insights
    print("\nTop Insights:")
    for insight in signals.get('insights', [])[:5]:  # Show top 5 insights
        print(f"- {insight.get('source', 'unknown')}: {insight.get('type', 'unknown')}")
        print(f"  {insight.get('interpretation', 'No interpretation available')}")
        print()
    
    # Check health status
    print("\nHealth Status:")
    health = alt_data.get_health_status()
    print(f"Overall status: {health['status']}")
    print(f"Source count: {health['source_count']}")
    
    # Example: Export signals to JSON for further analysis
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"alternative_data_signals_{today.strftime('%Y%m%d')}.json")
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)
    
    print(f"\nSignals exported to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
