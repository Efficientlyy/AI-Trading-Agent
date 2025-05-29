"""
Supply Chain Data Analysis for trading insights.

This module integrates with supply chain and logistics data providers to analyze
trends relevant for trading decision-making, such as:
- Shipping container movements
- Port congestion metrics
- Freight rates
- Manufacturing activity indicators
- Inventory levels
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import aiohttp
import numpy as np

from .base import AlternativeDataSource, AlternativeDataConfig
from ...health.monitoring import register_health_check, HealthStatus


logger = logging.getLogger(__name__)


class SupplyChainDataAnalyzer(AlternativeDataSource):
    """
    Analyzer for supply chain and logistics data to extract trading insights.
    
    This class connects to logistics data providers, processes shipping and
    inventory data, and extracts trading-relevant metrics.
    """
    
    SUPPORTED_PROVIDERS = {
        "freightos": "https://api.freightos.com/v1/",
        "flexport": "https://api.flexport.com/v2/",
        "container_xchange": "https://api.container-xchange.com/v1/",
        "searates": "https://api.searates.com/v1/"
    }
    
    SUPPORTED_DATA_TYPES = [
        "container_prices",
        "port_congestion",
        "freight_rates",
        "manufacturing_activity",
        "inventory_levels"
    ]
    
    # Key trade routes for shipping
    TRADE_ROUTES = [
        "china_to_us_west",
        "china_to_us_east",
        "china_to_europe",
        "us_to_europe",
        "asia_to_middle_east"
    ]
    
    def __init__(self, config: AlternativeDataConfig, provider: str = "freightos"):
        """
        Initialize the supply chain data analyzer.
        
        Args:
            config: Configuration with API keys and settings
            provider: Name of the supply chain data provider to use
        """
        self.provider = provider
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. "
                          f"Must be one of {list(self.SUPPORTED_PROVIDERS.keys())}")
        
        # Set the endpoint based on the provider
        if not config.endpoint:
            config.endpoint = self.SUPPORTED_PROVIDERS[provider]
            
        super().__init__(config)
        
        # Register with health monitoring system
        register_health_check(
            component_id=f"supply_chain_{provider}",
            check_function=self.get_health_status,
            interval_seconds=300
        )
    
    def _initialize(self) -> None:
        """Initialize the connection to the supply chain data provider."""
        self.session = None
        self.connected = False
        self.data_cache = {}
        logger.info(f"Initialized supply chain data analyzer with provider: {self.provider}")
    
    async def _ensure_session(self) -> None:
        """Ensure an active HTTP session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
    
    async def _close_session(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def fetch_data(self, query: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """
        Fetch supply chain data based on query parameters.
        
        Args:
            query: Dictionary containing query parameters such as:
                - data_type: Type of supply chain data to retrieve
                - routes: Specific trade routes to analyze
                - start_date: Beginning of time period
                - end_date: End of time period
            **kwargs: Additional provider-specific parameters
            
        Returns:
            DataFrame containing supply chain metrics
        """
        data_type = query.get("data_type")
        if data_type not in self.SUPPORTED_DATA_TYPES:
            raise ValueError(f"Unsupported data type: {data_type}. "
                          f"Must be one of {self.SUPPORTED_DATA_TYPES}")
        
        routes = query.get("routes", self.TRADE_ROUTES)
        
        # Create cache key
        cache_key = str(hash(str({
            "data_type": data_type,
            "routes": sorted(routes),
            "start_date": query.get("start_date"),
            "end_date": query.get("end_date")
        })))
        
        # Check cache
        if cache_key in self._cache:
            logger.info(f"Returning cached results for query: {cache_key}")
            return self._cache[cache_key]
        
        await self._ensure_session()
        
        try:
            # In a real implementation, this would make real API calls
            # For now, we'll generate mock data
            
            results = self._generate_mock_data(data_type, routes, query)
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Cache the results
            self._cache[cache_key] = df
            self.last_updated = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching supply chain data: {e}")
            self.connected = False
            raise
    
    def _generate_mock_data(
        self, 
        data_type: str, 
        routes: List[str],
        query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate mock supply chain data for testing.
        
        Args:
            data_type: Type of data to generate
            routes: Trade routes to include
            query: Original query parameters
            
        Returns:
            List of data points
        """
        start_date = query.get("start_date", datetime.now() - timedelta(days=30))
        end_date = query.get("end_date", datetime.now())
        
        # Convert strings to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Generate dates between start and end date
        total_days = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(total_days + 1)]
        
        results = []
        
        for route in routes:
            baseline, volatility = self._get_route_parameters(route, data_type)
            trend = np.random.choice([-0.2, -0.1, 0, 0.1, 0.2])  # Random trend direction
            
            for date in dates:
                # Calculate day offset for trend calculation
                day_offset = (date - start_date).days / total_days
                
                # Generate value with trend and random noise
                base_value = baseline * (1 + trend * day_offset)
                noise = np.random.normal(0, volatility * baseline)
                value = max(0, base_value + noise)  # Ensure non-negative
                
                # Add seasonal patterns for certain data types
                if data_type in ["freight_rates", "port_congestion"]:
                    # Add weekly patterns (e.g., weekend effects)
                    day_of_week = date.weekday()
                    if day_of_week >= 5:  # Weekend
                        value *= 0.9  # Less activity on weekends
                
                # Add result
                data_point = {
                    "date": date.isoformat(),
                    "route": route,
                    "data_type": data_type,
                    "value": value
                }
                
                # Add specific fields based on data type
                if data_type == "container_prices":
                    data_point["container_type"] = "40ft_standard"
                    data_point["unit"] = "USD"
                    data_point["source_port"] = route.split("_to_")[0]
                    data_point["destination_port"] = route.split("_to_")[1]
                
                elif data_type == "port_congestion":
                    data_point["unit"] = "days"
                    data_point["port"] = route.split("_to_")[1]
                    data_point["vessel_count"] = int(value * 5)  # Ships waiting
                
                elif data_type == "freight_rates":
                    data_point["unit"] = "USD/container"
                    data_point["source_port"] = route.split("_to_")[0]
                    data_point["destination_port"] = route.split("_to_")[1]
                
                elif data_type == "manufacturing_activity":
                    data_point["unit"] = "index"
                    data_point["region"] = route.split("_to_")[0]
                    data_point["sector"] = np.random.choice([
                        "electronics", "automotive", "consumer_goods", "machinery"
                    ])
                
                elif data_type == "inventory_levels":
                    data_point["unit"] = "days_of_supply"
                    data_point["region"] = route.split("_to_")[1]
                    data_point["sector"] = np.random.choice([
                        "retail", "automotive", "electronics", "apparel"
                    ])
                
                results.append(data_point)
        
        return results
    
    def _get_route_parameters(self, route: str, data_type: str) -> tuple:
        """
        Get baseline value and volatility for a specific route and data type.
        
        Args:
            route: Trade route
            data_type: Type of data
            
        Returns:
            Tuple of (baseline_value, volatility)
        """
        # Define realistic baseline values for each data type and route
        baselines = {
            "container_prices": {
                "china_to_us_west": 2500,
                "china_to_us_east": 3200,
                "china_to_europe": 2800,
                "us_to_europe": 1800,
                "asia_to_middle_east": 1500
            },
            "port_congestion": {
                "china_to_us_west": 5,  # Days of delay
                "china_to_us_east": 4,
                "china_to_europe": 3,
                "us_to_europe": 2,
                "asia_to_middle_east": 3
            },
            "freight_rates": {
                "china_to_us_west": 5500,
                "china_to_us_east": 6500,
                "china_to_europe": 5000,
                "us_to_europe": 3500,
                "asia_to_middle_east": 3000
            },
            "manufacturing_activity": {
                "china_to_us_west": 52,  # Index value
                "china_to_us_east": 52,
                "china_to_europe": 52,
                "us_to_europe": 50,
                "asia_to_middle_east": 51
            },
            "inventory_levels": {
                "china_to_us_west": 30,  # Days of supply
                "china_to_us_east": 30,
                "china_to_europe": 28,
                "us_to_europe": 32,
                "asia_to_middle_east": 35
            }
        }
        
        # Define volatility for each data type
        volatility = {
            "container_prices": 0.08,
            "port_congestion": 0.15,
            "freight_rates": 0.1,
            "manufacturing_activity": 0.03,
            "inventory_levels": 0.05
        }
        
        return baselines[data_type][route], volatility[data_type]
    
    def process_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process raw supply chain data into trading signals.
        
        Args:
            data: DataFrame with raw analysis results
            
        Returns:
            Dictionary with processed signals
        """
        if data.empty:
            return {"signal": "neutral", "strength": 0, "insights": []}
        
        # Convert date strings to datetime if needed
        if "date" in data.columns and isinstance(data["date"].iloc[0], str):
            data["date"] = pd.to_datetime(data["date"])
        
        # Group by data_type
        signals = []
        insights = []
        
        for data_type, group in data.groupby("data_type"):
            if data_type == "container_prices":
                insight, signal = self._process_container_prices(group)
            elif data_type == "port_congestion":
                insight, signal = self._process_port_congestion(group)
            elif data_type == "freight_rates":
                insight, signal = self._process_freight_rates(group)
            elif data_type == "manufacturing_activity":
                insight, signal = self._process_manufacturing_activity(group)
            elif data_type == "inventory_levels":
                insight, signal = self._process_inventory_levels(group)
            else:
                continue
                
            if insight:
                insights.append(insight)
            if signal:
                signals.append(signal)
        
        if not signals:
            return {"signal": "neutral", "strength": 0, "insights": []}
        
        # Aggregate signals
        signal_strengths = [s["strength"] * (1 if s["direction"] == "bullish" else -1) for s in signals]
        avg_strength = sum(signal_strengths) / len(signal_strengths)
        
        if avg_strength > 0:
            signal = "bullish"
            strength = avg_strength
        elif avg_strength < 0:
            signal = "bearish"
            strength = abs(avg_strength)
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "signal": signal,
            "strength": float(strength),
            "insights": insights,
            "raw_signals": signals,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_container_prices(self, data: pd.DataFrame) -> tuple:
        """Process container price data."""
        # Calculate trends
        data = data.sort_values("date")
        recent_data = data.iloc[-7:]  # Last week
        
        # Get average price
        avg_price = recent_data["value"].mean()
        
        # Calculate trend
        if len(data) > 14:
            previous_period = data.iloc[-14:-7]["value"].mean()
            change_pct = (avg_price / previous_period - 1) * 100
            
            # Interpret trend
            if change_pct > 5:
                trend_desc = "increasing rapidly"
                interpretation = "Container prices are rising sharply, indicating strong demand or limited supply. This often precedes inventory building and may impact retail and manufacturing sectors."
                signal_direction = "bearish"  # Higher shipping costs are typically bearish for retail/manufacturing
                signal_strength = min(0.3 + (change_pct - 5) * 0.02, 0.7)  # Cap at 0.7
            elif change_pct > 2:
                trend_desc = "increasing"
                interpretation = "Container prices are rising, suggesting growing demand for shipping capacity. Watch for impacts on retail margins and manufacturing costs."
                signal_direction = "bearish"
                signal_strength = 0.2
            elif change_pct < -5:
                trend_desc = "decreasing rapidly"
                interpretation = "Container prices are falling sharply, potentially indicating weakening global trade or increased shipping capacity. May be positive for companies with high shipping costs."
                signal_direction = "bullish"  # Lower shipping costs are typically bullish for retail/manufacturing
                signal_strength = min(0.3 + (abs(change_pct) - 5) * 0.02, 0.7)
            elif change_pct < -2:
                trend_desc = "decreasing"
                interpretation = "Container prices are declining, which may reduce cost pressures for importers."
                signal_direction = "bullish"
                signal_strength = 0.2
            else:
                trend_desc = "stable"
                interpretation = "Container prices are stable, suggesting balanced supply and demand in shipping."
                signal_direction = "neutral"
                signal_strength = 0
        else:
            trend_desc = "insufficient data"
            interpretation = "Not enough historical data to determine container price trends."
            signal_direction = "neutral"
            signal_strength = 0
            change_pct = 0
        
        insight = {
            "type": "container_prices",
            "average_price": float(avg_price),
            "change_percentage": float(change_pct) if 'change_pct' in locals() else None,
            "trend": trend_desc,
            "interpretation": interpretation
        }
        
        signal = {
            "direction": signal_direction,
            "strength": float(signal_strength),
            "source": "container_prices"
        }
        
        return insight, signal
    
    def _process_port_congestion(self, data: pd.DataFrame) -> tuple:
        """Process port congestion data."""
        # Calculate average congestion
        recent_data = data.sort_values("date").iloc[-7:]  # Last week
        avg_congestion = recent_data["value"].mean()
        
        # Determine congestion level
        if avg_congestion > 7:
            level_desc = "severe"
            interpretation = "Severe port congestion indicates significant supply chain disruptions. Expect inventory shortages and delayed shipments affecting retail and manufacturing."
            signal_direction = "bearish"
            signal_strength = 0.6
        elif avg_congestion > 4:
            level_desc = "high"
            interpretation = "High port congestion suggests ongoing supply chain challenges. Watch for impacts on just-in-time manufacturing and retail inventory levels."
            signal_direction = "bearish"
            signal_strength = 0.4
        elif avg_congestion > 2:
            level_desc = "moderate"
            interpretation = "Moderate port congestion indicates typical operation with some delays. Limited impact on supply chains."
            signal_direction = "neutral"
            signal_strength = 0.1
        else:
            level_desc = "low"
            interpretation = "Low port congestion suggests smooth supply chain operations. Favorable for timely inventory replenishment."
            signal_direction = "bullish"
            signal_strength = 0.2
        
        # Calculate trend if enough data
        trend_desc = "undetermined"
        if len(data) > 14:
            previous_period = data.iloc[-14:-7]["value"].mean()
            change = avg_congestion - previous_period
            
            if change > 1:
                trend_desc = "worsening"
                # Intensify bearish signal if congestion is worsening
                if signal_direction == "bearish":
                    signal_strength = min(signal_strength + 0.2, 0.8)
            elif change < -1:
                trend_desc = "improving"
                # Reduce bearish signal or strengthen bullish signal if improving
                if signal_direction == "bearish":
                    signal_strength = max(signal_strength - 0.2, 0.1)
                else:
                    signal_strength = min(signal_strength + 0.2, 0.7)
            else:
                trend_desc = "stable"
        
        insight = {
            "type": "port_congestion",
            "average_delay_days": float(avg_congestion),
            "congestion_level": level_desc,
            "trend": trend_desc,
            "interpretation": interpretation
        }
        
        signal = {
            "direction": signal_direction,
            "strength": float(signal_strength),
            "source": "port_congestion"
        }
        
        return insight, signal
    
    def _process_freight_rates(self, data: pd.DataFrame) -> tuple:
        """Process freight rate data."""
        # Similar to container prices, but with different thresholds
        data = data.sort_values("date")
        recent_data = data.iloc[-7:]  # Last week
        avg_rate = recent_data["value"].mean()
        
        # Calculate trend
        if len(data) > 14:
            previous_period = data.iloc[-14:-7]["value"].mean()
            change_pct = (avg_rate / previous_period - 1) * 100
            
            # Interpret trend
            if change_pct > 10:
                trend_desc = "increasing rapidly"
                interpretation = "Freight rates are surging, suggesting capacity constraints or strong demand. Negative for margins of import-dependent businesses."
                signal_direction = "bearish"
                signal_strength = 0.5
            elif change_pct > 3:
                trend_desc = "increasing"
                interpretation = "Freight rates are rising, indicating growing shipping demand. May pressure margins for retailers and manufacturers."
                signal_direction = "bearish"
                signal_strength = 0.3
            elif change_pct < -10:
                trend_desc = "decreasing rapidly"
                interpretation = "Freight rates are dropping sharply, potentially indicating weakening demand or increased capacity. Positive for importers but may signal economic slowdown."
                signal_direction = "mixed"
                signal_strength = 0.2
            elif change_pct < -3:
                trend_desc = "decreasing"
                interpretation = "Freight rates are declining, which may improve margins for importers."
                signal_direction = "bullish"
                signal_strength = 0.2
            else:
                trend_desc = "stable"
                interpretation = "Freight rates are stable, suggesting balanced shipping market conditions."
                signal_direction = "neutral"
                signal_strength = 0
        else:
            trend_desc = "insufficient data"
            interpretation = "Not enough historical data to determine freight rate trends."
            signal_direction = "neutral"
            signal_strength = 0
            change_pct = 0
        
        insight = {
            "type": "freight_rates",
            "average_rate": float(avg_rate),
            "change_percentage": float(change_pct) if 'change_pct' in locals() else None,
            "trend": trend_desc,
            "interpretation": interpretation
        }
        
        signal = {
            "direction": signal_direction,
            "strength": float(signal_strength),
            "source": "freight_rates"
        }
        
        return insight, signal
    
    def _process_manufacturing_activity(self, data: pd.DataFrame) -> tuple:
        """Process manufacturing activity data."""
        recent_data = data.sort_values("date").iloc[-7:]
        avg_index = recent_data["value"].mean()
        
        # Interpret level (typically PMI-like index where >50 = expansion)
        if avg_index > 55:
            level_desc = "strong expansion"
            base_interpretation = "Manufacturing activity shows strong expansion, indicating robust economic growth. Positive for industrial sectors and raw materials."
            base_direction = "bullish"
            base_strength = 0.6
        elif avg_index > 50:
            level_desc = "moderate expansion"
            base_interpretation = "Manufacturing activity is expanding at a moderate pace, suggesting steady economic growth."
            base_direction = "bullish"
            base_strength = 0.3
        elif avg_index > 48:
            level_desc = "stagnation"
            base_interpretation = "Manufacturing activity is near the neutral threshold, indicating economic uncertainty."
            base_direction = "neutral"
            base_strength = 0.1
        elif avg_index > 45:
            level_desc = "mild contraction"
            base_interpretation = "Manufacturing activity is contracting mildly. Caution advised for industrial sectors."
            base_direction = "bearish"
            base_strength = 0.3
        else:
            level_desc = "significant contraction"
            base_interpretation = "Manufacturing activity shows significant contraction, suggesting economic downturn. Negative for industrial sectors and cyclicals."
            base_direction = "bearish"
            base_strength = 0.6
        
        # Calculate trend if enough data
        if len(data) > 14:
            previous_period = data.iloc[-14:-7]["value"].mean()
            change = avg_index - previous_period
            
            if change > 2:
                trend_desc = "improving rapidly"
                interpretation = f"{base_interpretation} The rapid improvement suggests accelerating growth momentum."
                signal_strength = min(base_strength + 0.2, 0.8)
            elif change > 0.5:
                trend_desc = "improving"
                interpretation = f"{base_interpretation} The positive trend indicates building momentum."
                signal_strength = min(base_strength + 0.1, 0.7)
            elif change < -2:
                trend_desc = "deteriorating rapidly"
                interpretation = f"{base_interpretation} The rapid deterioration signals significant weakening in industrial activity."
                # Flip direction if crossing the 50 threshold
                if avg_index < 50 and previous_period > 50:
                    signal_direction = "bearish"
                    signal_strength = 0.6
                else:
                    signal_direction = base_direction
                    signal_strength = base_strength + 0.2
            elif change < -0.5:
                trend_desc = "deteriorating"
                interpretation = f"{base_interpretation} The negative trend suggests weakening conditions."
                signal_direction = base_direction
                signal_strength = base_strength
            else:
                trend_desc = "stable"
                interpretation = f"{base_interpretation} The stable trend suggests consistent conditions."
                signal_direction = base_direction
                signal_strength = base_strength
        else:
            trend_desc = "insufficient data"
            interpretation = base_interpretation
            signal_direction = base_direction
            signal_strength = base_strength
        
        insight = {
            "type": "manufacturing_activity",
            "average_index": float(avg_index),
            "activity_level": level_desc,
            "trend": trend_desc,
            "interpretation": interpretation
        }
        
        signal = {
            "direction": signal_direction if 'signal_direction' in locals() else base_direction,
            "strength": float(signal_strength),
            "source": "manufacturing_activity"
        }
        
        return insight, signal
    
    def _process_inventory_levels(self, data: pd.DataFrame) -> tuple:
        """Process inventory levels data."""
        recent_data = data.sort_values("date").iloc[-7:]
        avg_inventory = recent_data["value"].mean()  # Days of supply
        
        # Interpret inventory levels
        if avg_inventory > 45:
            level_desc = "excessive"
            base_interpretation = "Inventory levels are excessive, suggesting weak demand or overproduction. Negative for manufacturing in the near term as production may slow."
            base_direction = "bearish"
            base_strength = 0.5
        elif avg_inventory > 35:
            level_desc = "high"
            base_interpretation = "Inventory levels are high, potentially indicating softening demand. Watch for inventory corrections affecting production."
            base_direction = "bearish"
            base_strength = 0.3
        elif avg_inventory > 25:
            level_desc = "balanced"
            base_interpretation = "Inventory levels are balanced, suggesting healthy supply chain functioning."
            base_direction = "neutral"
            base_strength = 0
        elif avg_inventory > 15:
            level_desc = "lean"
            base_interpretation = "Inventory levels are lean, indicating strong demand or supply constraints. Positive for future production increases."
            base_direction = "bullish"
            base_strength = 0.3
        else:
            level_desc = "critically low"
            base_interpretation = "Inventory levels are critically low, suggesting either extremely strong demand or severe supply chain disruptions. Expect production increases but possible stockouts."
            base_direction = "bullish"
            base_strength = 0.5
        
        # Calculate trend if enough data
        if len(data) > 14:
            previous_period = data.iloc[-14:-7]["value"].mean()
            change = avg_inventory - previous_period
            
            if change > 5:
                trend_desc = "building rapidly"
                interpretation = f"{base_interpretation} The rapid inventory build may indicate weakening sales or increased production capacity."
                if level_desc in ["excessive", "high"]:
                    signal_direction = "bearish"
                    signal_strength = min(base_strength + 0.2, 0.7)
                else:
                    signal_direction = base_direction
                    signal_strength = base_strength
            elif change > 2:
                trend_desc = "building"
                interpretation = f"{base_interpretation} The inventory build indicates production exceeding demand."
                signal_direction = base_direction
                signal_strength = base_strength
            elif change < -5:
                trend_desc = "depleting rapidly"
                interpretation = f"{base_interpretation} The rapid inventory depletion suggests strong demand or supply chain challenges."
                if level_desc in ["critically low", "lean"]:
                    signal_direction = "bullish"
                    signal_strength = min(base_strength + 0.2, 0.7)
                else:
                    signal_direction = base_direction
                    signal_strength = base_strength
            elif change < -2:
                trend_desc = "depleting"
                interpretation = f"{base_interpretation} The inventory drawdown indicates demand exceeding production."
                signal_direction = base_direction
                signal_strength = base_strength
            else:
                trend_desc = "stable"
                interpretation = f"{base_interpretation} The stable inventory levels suggest balanced supply and demand."
                signal_direction = base_direction
                signal_strength = base_strength
        else:
            trend_desc = "insufficient data"
            interpretation = base_interpretation
            signal_direction = base_direction
            signal_strength = base_strength
        
        insight = {
            "type": "inventory_levels",
            "average_days_supply": float(avg_inventory),
            "inventory_level": level_desc,
            "trend": trend_desc,
            "interpretation": interpretation
        }
        
        signal = {
            "direction": signal_direction if 'signal_direction' in locals() else base_direction,
            "strength": float(signal_strength),
            "source": "inventory_levels"
        }
        
        return insight, signal
    
    def _is_connected(self) -> bool:
        """Check if connected to the supply chain data provider."""
        return self.connected
    
    async def __aenter__(self):
        """Support for async context manager."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        await self._close_session()
