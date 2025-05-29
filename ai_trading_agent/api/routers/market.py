"""
Market Data API endpoints for internal frontend use.

This module provides API endpoints for the frontend to access 
real-time and historical market data from the MarketDataProvider.
"""

import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel

# Import MarketDataProvider
from ai_trading_agent.data_providers.crypto.provider import MarketDataProvider

# Import models
from .market_models import (
    Asset, AssetsResponse, AssetDetailResponse, HistoricalBar,
    HistoricalDataRequest, HistoricalDataResponse, TimeFrame,
    MarketStatus, TechnicalIndicatorsResponse, MarketOverviewResponse,
    MarketRegime, VolatilityRegime
)

# Import utility functions
from .market_utils import generate_mock_historical_data, detect_market_regime, get_technical_indicators

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/market", tags=["Market Data"])

# Initialize the market data provider
market_data_provider = MarketDataProvider()

# Create startup and shutdown functions that will be registered with the main app
async def startup_market_data():
    """Start the market data provider when the API starts."""
    logger.info("Starting market data provider...")
    market_data_provider.start()
    # Subscribe to common cryptocurrency pairs
    market_data_provider.subscribe("BTC/USD")
    market_data_provider.subscribe("ETH/USD")
    logger.info("Market data provider started successfully")

async def shutdown_market_data():
    """Stop the market data provider when the API shuts down."""
    logger.info("Stopping market data provider...")
    market_data_provider.stop()
    logger.info("Market data provider stopped successfully")

# We now use imported models from market_models.py

# Asset symbols to names mapping, including detailed information
ASSET_INFO = {
    "BTC/USD": {
        "name": "Bitcoin",
        "symbol": "BTC",
        "type": "crypto",
        "icon": "btc-icon",
        "color": "#F7931A",
        "description": "The original cryptocurrency and largest by market capitalization",
        "volatility": "high"
    },
    "ETH/USD": {
        "name": "Ethereum",
        "symbol": "ETH",
        "type": "crypto",
        "icon": "eth-icon",
        "color": "#627EEA",
        "description": "Smart contract platform enabling decentralized applications",
        "volatility": "high"
    },
    "XRP/USD": {
        "name": "Ripple",
        "symbol": "XRP",
        "type": "crypto",
        "icon": "xrp-icon",
        "color": "#23292F",
        "description": "Digital payment protocol and cryptocurrency",
        "volatility": "medium"
    },
    "ADA/USD": {
        "name": "Cardano",
        "symbol": "ADA",
        "type": "crypto",
        "icon": "ada-icon",
        "color": "#0033AD",
        "description": "Proof-of-stake blockchain platform with smart contract capabilities",
        "volatility": "high"
    },
    "SOL/USD": {
        "name": "Solana",
        "symbol": "SOL",
        "type": "crypto",
        "icon": "sol-icon",
        "color": "#00FFA3",
        "description": "High-performance blockchain supporting smart contracts and DeFi",
        "volatility": "very-high"
    },
    "DOGE/USD": {
        "name": "Dogecoin",
        "symbol": "DOGE",
        "type": "crypto",
        "icon": "doge-icon",
        "color": "#C2A633",
        "description": "Cryptocurrency that started as a meme, known for its Shiba Inu logo",
        "volatility": "very-high"
    }
}

# Extract just the names for backward compatibility
ASSET_NAMES = {symbol: info["name"] for symbol, info in ASSET_INFO.items()}

@router.get("/assets", response_model=AssetsResponse)
async def get_assets():
    """
    Get a list of available trading assets with real-time prices.
    
    This endpoint returns a list of available trading assets with 
    current prices and other metadata.
    
    Returns:
        List of assets with current market data
    """
    try:
        # Create a list of asset info with real-time prices
        assets = []
        
        # Include all cryptocurrencies defined in ASSET_INFO
        crypto_symbols = list(ASSET_INFO.keys())
        
        for symbol in crypto_symbols:
            try:
                # Get current price
                price = market_data_provider.get_current_price(symbol)
                
                # Only include assets where we have a price
                if price is not None:
                    # Get the source of this price, if available
                    source = market_data_provider.data_source.get(symbol, "Unknown") if hasattr(market_data_provider, 'data_source') else "Unknown"
                    
                    # Extract confidence score if available in the source string
                    confidence = None
                    if '(' in source and '%' in source:
                        try:
                            confidence_str = source.split('(')[1].split('%')[0]
                            confidence = int(confidence_str)
                        except:
                            confidence = None
                    
                    # Generate a random 24h change for demo purposes
                    # In production, this would be calculated from historical data
                    change_24h = random.uniform(-5.0, 7.0) if symbol not in ["BTC/USD", "ETH/USD"] else random.uniform(-3.0, 5.0)
                    
                    # Generate realistic volume data
                    volume_24h = None
                    if symbol == "BTC/USD":
                        volume_24h = random.uniform(20000000000, 30000000000)  # $20-30B
                    elif symbol == "ETH/USD":
                        volume_24h = random.uniform(8000000000, 15000000000)  # $8-15B
                    else:
                        volume_24h = random.uniform(500000000, 5000000000)  # $500M-5B
                    
                    # Get asset info
                    asset_info = ASSET_INFO.get(symbol, {})
                    
                    # Create asset object with enhanced info
                    asset = Asset(
                        symbol=symbol,
                        name=asset_info.get("name", symbol),
                        type="crypto",
                        price=price,
                        change_24h=change_24h,
                        volume_24h=volume_24h,
                        source=source.split('(')[0].strip() if '(' in source else source,
                        confidence=confidence,
                        description=asset_info.get("description"),
                        color=asset_info.get("color"),
                        icon=asset_info.get("icon"),
                        volatility=asset_info.get("volatility"),
                        high_24h=price * (1 + random.uniform(0.01, 0.05)),
                        low_24h=price * (1 - random.uniform(0.01, 0.05)),
                        current_regime=MarketRegime.BULL if change_24h > 0 else MarketRegime.BEAR,
                        volatility_regime=VolatilityRegime.HIGH if asset_info.get("volatility") == "high" else VolatilityRegime.MODERATE
                    )
                    assets.append(asset)
            except Exception as e:
                logger.warning(f"Error getting data for {symbol}: {e}")
                continue
        
        # Return the assets
        return AssetsResponse(assets=assets)
    except Exception as e:
        logger.error(f"Error getting assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get assets: {str(e)}"
        )

@router.get("/price/{symbol}")
async def get_price(symbol: str = Path(..., description="Trading symbol (e.g., 'BTC/USD')")):
    """
    Get the current price for a trading symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Current price information
    """
    try:
        # Get current price
        price = market_data_provider.get_current_price(symbol)
        
        if price is None:
            raise HTTPException(
                status_code=404,
                detail=f"Price not available for {symbol}"
            )
        
        # Get source information if available
        source = market_data_provider.data_source.get(symbol, "Unknown") if hasattr(market_data_provider, 'data_source') else "Unknown"
        
        # Return price info
        return {
            "symbol": symbol,
            "price": price,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get price: {str(e)}"
        )

@router.post("/historical", response_model=HistoricalDataResponse)
async def get_historical_data(request: HistoricalDataRequest):
    """
    Get historical price data for a trading symbol.
    
    This endpoint returns historical OHLCV data for a specified symbol and timeframe.
    If real historical data is available from our provider, it will be returned.
    Otherwise, synthetic data will be generated based on current price and market characteristics.
    
    Args:
        request: Request parameters including symbol, timeframe, limit, etc.
        
    Returns:
        Historical price data
    """
    try:
        symbol = request.symbol
        timeframe = request.timeframe
        limit = request.limit
        start_date = request.start_date
        end_date = request.end_date
        
        logger.info(f"Fetching historical data for {symbol} at {timeframe} timeframe, limit={limit}")
        
        try:
            # Try to get historical data from the market data provider
            # This is not implemented yet in MarketDataProvider, so we'll use synthetic data
            raise NotImplementedError("Real historical data not yet available")
        except Exception as e:
            logger.info(f"Falling back to synthetic historical data: {e}")
            # Generate mock historical data
            data = generate_mock_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                start_date=start_date,
                end_date=end_date
            )
        
        # Return historical data
        return HistoricalDataResponse(
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            currency="USD" if "/USD" in symbol else symbol.split("/")[1]
        )
    except Exception as e:
        logger.error(f"Error getting historical data for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get historical data: {str(e)}"
        )

@router.get("/asset/{symbol}", response_model=AssetDetailResponse)
async def get_asset_detail(symbol: str = Path(..., description="Trading symbol (e.g., 'BTC/USD')")):
    """
    Get detailed information for a specific asset including price, historical data, and technical indicators.
    
    This endpoint provides comprehensive information about a trading asset including:
    - Current price and market status
    - Recent historical data
    - Technical indicators
    - Related assets
    - Market sentiment
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Detailed asset information
    """
    try:
        # Get current price
        price = market_data_provider.get_current_price(symbol)
        
        if price is None:
            raise HTTPException(
                status_code=404,
                detail=f"Asset not available: {symbol}"
            )
        
        # Get asset info from our mapping
        asset_info = ASSET_INFO.get(symbol, {})
        
        # Get source information if available
        source = market_data_provider.data_source.get(symbol, "Unknown") if hasattr(market_data_provider, 'data_source') else "Unknown"
        
        # Extract confidence score if available in the source string
        confidence = None
        if '(' in source and '%' in source:
            try:
                confidence_str = source.split('(')[1].split('%')[0]
                confidence = int(confidence_str)
            except:
                confidence = None
        
        # Generate random metrics for demo purposes
        change_24h = random.uniform(-5.0, 7.0) if symbol not in ["BTC/USD", "ETH/USD"] else random.uniform(-3.0, 5.0)
        
        # Generate realistic volume
        volume_24h = None
        if symbol == "BTC/USD":
            volume_24h = random.uniform(20000000000, 30000000000)  # $20-30B
        elif symbol == "ETH/USD":
            volume_24h = random.uniform(8000000000, 15000000000)  # $8-15B
        else:
            volume_24h = random.uniform(500000000, 5000000000)  # $500M-5B
        
        # Get historical data for the last 24 hours (hourly bars)
        historical_data = generate_mock_historical_data(
            symbol=symbol,
            timeframe=TimeFrame.HOUR_1,
            limit=24,
            end_date=datetime.now()
        )
        
        # Detect market regime
        market_regime, volatility_regime = detect_market_regime(historical_data)
        
        # Calculate technical indicators
        tech_indicators = get_technical_indicators(historical_data)
        
        # Create the asset object
        asset = Asset(
            symbol=symbol,
            name=asset_info.get("name", symbol),
            type="crypto",
            price=price,
            change_24h=change_24h,
            volume_24h=volume_24h,
            source=source.split('(')[0].strip() if '(' in source else source,
            confidence=confidence,
            description=asset_info.get("description"),
            color=asset_info.get("color"),
            icon=asset_info.get("icon"),
            volatility=asset_info.get("volatility"),
            high_24h=price * (1 + random.uniform(0.01, 0.05)),
            low_24h=price * (1 - random.uniform(0.01, 0.05)),
            current_regime=market_regime,
            volatility_regime=volatility_regime,
            market_cap=price * 1000000 * random.uniform(10, 1000) if "BTC" not in symbol else price * 19000000
        )
        
        # Get related assets (e.g., for BTC, related assets might be ETH, etc.)
        related_assets = []
        for rel_symbol in list(ASSET_INFO.keys())[:3]:
            if rel_symbol != symbol:
                rel_price = market_data_provider.get_current_price(rel_symbol) or 0.0
                if rel_price > 0:
                    rel_asset_info = ASSET_INFO.get(rel_symbol, {})
                    rel_asset = Asset(
                        symbol=rel_symbol,
                        name=rel_asset_info.get("name", rel_symbol),
                        type="crypto",
                        price=rel_price,
                        change_24h=random.uniform(-5.0, 7.0),
                        volume_24h=None,
                        color=rel_asset_info.get("color"),
                        icon=rel_asset_info.get("icon")
                    )
                    related_assets.append(rel_asset)
        
        # Generate market sentiment (0-100 scale)
        market_sentiment = 50 + (change_24h * 5)  # Higher if price is increasing
        market_sentiment = max(0, min(100, market_sentiment))  # Clamp to 0-100
        
        # Determine market status (always open for crypto)
        market_status = MarketStatus.OPEN
        
        # Create the response
        return AssetDetailResponse(
            asset=asset,
            market_status=market_status,
            last_updated=datetime.now(),
            historical_data=historical_data,
            related_assets=related_assets,
            market_sentiment=market_sentiment,
            technical_indicators=tech_indicators
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting asset details for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get asset details: {str(e)}"
        )

@router.get("/overview", response_model=MarketOverviewResponse)
async def get_market_overview():
    """
    Get an overview of the cryptocurrency market.
    
    This endpoint provides a high-level overview of the market including:
    - Top gainers and losers
    - Most volatile assets
    - Overall market sentiment
    - Total trading volume
    
    Returns:
        Market overview data
    """
    try:
        assets = []
        
        # Get data for all assets
        for symbol in ASSET_INFO.keys():
            try:
                price = market_data_provider.get_current_price(symbol)
                if price is not None:
                    # Generate random metrics for demo purposes
                    change_24h = random.uniform(-8.0, 10.0)
                    
                    # Get asset info
                    asset_info = ASSET_INFO.get(symbol, {})
                    
                    # Create asset object
                    asset = Asset(
                        symbol=symbol,
                        name=asset_info.get("name", symbol),
                        type="crypto",
                        price=price,
                        change_24h=change_24h,
                        volume_24h=random.uniform(100000000, 30000000000),
                        color=asset_info.get("color"),
                        icon=asset_info.get("icon")
                    )
                    assets.append(asset)
            except Exception as e:
                logger.warning(f"Error getting data for {symbol}: {e}")
                continue
        
        # Sort assets for top gainers and losers
        assets_by_change = sorted(assets, key=lambda x: x.change_24h or 0.0, reverse=True)
        top_gainers = assets_by_change[:3]
        top_losers = assets_by_change[-3:]
        
        # Get most volatile assets (using random data for now)
        most_volatile = sorted(assets, key=lambda x: abs(x.change_24h or 0.0), reverse=True)[:3]
        
        # Calculate overall market sentiment (0-100)
        # Average change_24h across assets, scaled to 0-100 range
        avg_change = sum(a.change_24h or 0.0 for a in assets) / max(1, len(assets))
        market_sentiment = 50 + (avg_change * 5)  # Center at 50, scale by 5
        market_sentiment = max(0, min(100, market_sentiment))  # Clamp to 0-100
        
        # Calculate total trading volume
        total_volume = sum(a.volume_24h or 0.0 for a in assets)
        
        # Return market overview
        return MarketOverviewResponse(
            timestamp=datetime.now(),
            market_status=MarketStatus.OPEN,  # Crypto markets are always open
            top_gainers=top_gainers,
            top_losers=top_losers,
            most_volatile=most_volatile,
            market_sentiment=market_sentiment,
            trading_volume=total_volume
        )
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market overview: {str(e)}"
        )
