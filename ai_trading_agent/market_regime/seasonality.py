"""
Seasonality Detection Module

This module provides tools for detecting seasonal patterns in market data,
which is an important component for temporal pattern recognition in market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats, signal
import logging

# Set up logger
logger = logging.getLogger(__name__)


class SeasonalityDetector:
    """
    Class for detecting seasonal patterns in financial time series data.
    
    Uses various methods for detecting seasonality including:
    - Autocorrelation analysis
    - Calendar-based patterns
    - Fourier analysis for cycle detection
    """
    
    def __init__(self, min_periods: int = 60):
        """
        Initialize the seasonality detector.
        
        Args:
            min_periods: Minimum number of data points needed for analysis
        """
        self.min_periods = min_periods
        self.seasonality_results = {}
    
    def detect_seasonality(self,
                          series: pd.Series,
                          asset_id: str = "default",
                          periods_to_test: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Detect seasonality in a time series.
        
        Args:
            series: Time series data to analyze
            asset_id: Identifier for the asset
            periods_to_test: List of potential seasonal periods to test
            
        Returns:
            Dictionary with seasonality analysis results
        """
        if series is None or len(series) < self.min_periods:
            logger.warning(f"Insufficient data for seasonality detection: {len(series) if series is not None else 0} points")
            return {
                "has_seasonality": False,
                "seasonal_periods": [],
                "strength": 0.0
            }
        
        # Default periods to test based on common financial cycles
        if periods_to_test is None:
            periods_to_test = self._get_default_periods(series)
        
        # Detect calendar-based patterns
        calendar_patterns = self._detect_calendar_patterns(series)
        
        # Detect autocorrelation-based seasonality
        acf_results = self._detect_autocorrelation_seasonality(series, periods_to_test)
        
        # Combine results
        has_seasonality = acf_results["has_seasonality"] or len(calendar_patterns) > 0
        
        results = {
            "has_seasonality": has_seasonality,
            "acf_results": acf_results,
            "calendar_patterns": calendar_patterns,
            "asset_id": asset_id,
            "timestamp": pd.Timestamp.now()
        }
        
        # Store results for this asset
        self.seasonality_results[asset_id] = results
        
        return results
    
    def _get_default_periods(self, series: pd.Series) -> List[int]:
        """
        Get default periods to test based on data characteristics.
        
        Args:
            series: Time series data
            
        Returns:
            List of default periods to test
        """
        # Try to infer frequency from the data
        if isinstance(series.index, pd.DatetimeIndex):
            freq = pd.infer_freq(series.index)
            
            if freq in ["D", "B"]:
                # Daily data - test for weekly, monthly, quarterly patterns
                return [5, 7, 20, 22, 30, 60, 90, 252]
            elif freq in ["W"]:
                # Weekly data - test for monthly, quarterly, yearly patterns
                return [4, 13, 26, 52]
            elif freq in ["M"]:
                # Monthly data - test for quarterly, yearly patterns
                return [3, 6, 12]
            elif freq and "H" in freq:
                # Hourly data - test for intraday, daily, weekly patterns
                return [4, 12, 24, 24*5, 24*7]
        
        # Default case - try various common periods
        return [5, 7, 20, 60, 252]
    
    def _detect_autocorrelation_seasonality(self, series: pd.Series, periods_to_test: List[int]) -> Dict[str, any]:
        """
        Detect seasonality using autocorrelation analysis.
        
        Args:
            series: Time series data
            periods_to_test: List of periods to test
            
        Returns:
            Dictionary with autocorrelation results
        """
        # Calculate autocorrelation
        max_lag = min(len(series) - 1, max(periods_to_test) * 2)
        
        try:
            # Use custom autocorrelation implementation to avoid statsmodels dependency
            series_values = series.values
            mean = np.mean(series_values)
            norm_series = series_values - mean
            acf_values = np.correlate(norm_series, norm_series, mode='full')
            acf_values = acf_values[len(acf_values)//2:] / np.sum(norm_series**2)
            acf_values = acf_values[:max_lag+1]
        except Exception as e:
            logger.warning(f"Error calculating ACF: {str(e)}")
            return {
                "has_seasonality": False,
                "seasonal_periods": [],
                "acf_values": []
            }
        
        # Find peaks in autocorrelation
        # A simple peak detection algorithm
        peaks = []
        for i in range(2, len(acf_values)-2):
            if (acf_values[i] > acf_values[i-1] and 
                acf_values[i] > acf_values[i-2] and
                acf_values[i] > acf_values[i+1] and
                acf_values[i] > acf_values[i+2] and
                acf_values[i] > 0.2):  # Significance threshold
                peaks.append(i)
        
        significant_periods = []
        for period in periods_to_test:
            if period < len(acf_values):
                acf_value = acf_values[period]
                
                # Check if this period corresponds to a significant peak
                if acf_value > 0.2:  # Threshold for significance
                    is_peak = period in peaks or any(abs(period - p) <= 1 for p in peaks)
                    if is_peak:
                        significant_periods.append({
                            "period": period,
                            "acf_value": float(acf_value),
                            "is_significant": True
                        })
        
        # Sort by ACF value
        significant_periods.sort(key=lambda x: x["acf_value"], reverse=True)
        
        return {
            "has_seasonality": len(significant_periods) > 0,
            "seasonal_periods": significant_periods,
            "acf_values": [float(x) for x in acf_values[:min(100, len(acf_values))]]
        }
    
    def _detect_calendar_patterns(self, series: pd.Series) -> Dict[str, float]:
        """
        Detect calendar-based seasonal patterns.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with calendar pattern results
        """
        # Check for day-of-week effects (only for daily data)
        calendar_effects = {}
        
        if not isinstance(series.index, pd.DatetimeIndex):
            return calendar_effects
            
        try:
            # Extract datetime components
            df = pd.DataFrame({'value': series})
            
            # Day of week effect
            df['day_of_week'] = series.index.dayofweek
            day_of_week_effect = df.groupby('day_of_week')['value'].mean()
            day_std = day_of_week_effect.std()
            day_mean = day_of_week_effect.mean()
            
            if day_mean != 0 and day_std / day_mean > 0.01:
                calendar_effects['day_of_week'] = float(day_std / day_mean)
                calendar_effects['day_of_week_values'] = {
                    day: float(value) for day, value in 
                    enumerate(day_of_week_effect.values)
                }
                
            # Month of year effect
            df['month'] = series.index.month
            month_effect = df.groupby('month')['value'].mean()
            month_std = month_effect.std()
            month_mean = month_effect.mean()
            
            if month_mean != 0 and month_std / month_mean > 0.01:
                calendar_effects['month_of_year'] = float(month_std / month_mean)
                calendar_effects['month_of_year_values'] = {
                    month: float(value) for month, value in 
                    enumerate(month_effect.values, 1)
                }
                
            # Quarter effect
            df['quarter'] = series.index.quarter
            quarter_effect = df.groupby('quarter')['value'].mean()
            quarter_std = quarter_effect.std()
            quarter_mean = quarter_effect.mean()
            
            if quarter_mean != 0 and quarter_std / quarter_mean > 0.01:
                calendar_effects['quarter'] = float(quarter_std / quarter_mean)
                calendar_effects['quarter_values'] = {
                    quarter: float(value) for quarter, value in 
                    enumerate(quarter_effect.values, 1)
                }
                
        except Exception as e:
            logger.warning(f"Error detecting calendar patterns: {str(e)}")
            
        return calendar_effects
        
    def get_seasonal_forecast(self, 
                             series: pd.Series,
                             asset_id: str = "default",
                             forecast_periods: int = 30) -> Dict[str, any]:
        """
        Generate a forecast based on detected seasonality patterns.
        
        Args:
            series: Time series data
            asset_id: Identifier for the asset
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast based on seasonality
        """
        # Check if we have seasonality results for this asset
        if asset_id not in self.seasonality_results:
            self.detect_seasonality(series, asset_id)
            
        if asset_id not in self.seasonality_results:
            return {
                "forecast": None,
                "has_seasonality": False
            }
            
        results = self.seasonality_results[asset_id]
        
        # If no seasonality, return empty forecast
        if not results["has_seasonality"]:
            return {
                "forecast": None,
                "has_seasonality": False
            }
            
        # Generate forecast based on detected seasonality
        forecast = {}
        
        # ACF-based seasonality
        acf_results = results["acf_results"]
        if acf_results["has_seasonality"]:
            # Get the most significant period
            best_period = acf_results["seasonal_periods"][0]["period"]
            
            # Create a basic seasonal forecast by repeating the pattern
            seasonal_pattern = []
            for i in range(best_period):
                if i < len(series):
                    start_idx = i
                    values = [series.values[j] for j in range(start_idx, len(series), best_period)]
                    seasonal_pattern.append(np.mean(values))
                else:
                    seasonal_pattern.append(np.mean(seasonal_pattern))
                    
            # Generate the forecast by repeating the pattern
            acf_forecast = []
            for i in range(forecast_periods):
                acf_forecast.append(seasonal_pattern[i % len(seasonal_pattern)])
                
            forecast["acf_forecast"] = acf_forecast
                
        # Calendar-based seasonality
        if "calendar_patterns" in results and results["calendar_patterns"]:
            calendar_patterns = results["calendar_patterns"]
            
            if isinstance(series.index, pd.DatetimeIndex):
                last_date = series.index[-1]
                dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_periods)]
                
                calendar_forecast = []
                
                for date in dates:
                    adjustment = 1.0
                    
                    # Day of week adjustment
                    if "day_of_week_values" in calendar_patterns:
                        day_values = calendar_patterns["day_of_week_values"]
                        day_of_week = date.dayofweek
                        if day_of_week in day_values:
                            day_adj = day_values[day_of_week] / np.mean(list(day_values.values()))
                            adjustment *= day_adj
                            
                    # Month adjustment
                    if "month_of_year_values" in calendar_patterns:
                        month_values = calendar_patterns["month_of_year_values"]
                        month = date.month
                        if month in month_values:
                            month_adj = month_values[month] / np.mean(list(month_values.values()))
                            adjustment *= month_adj
                            
                    # Apply adjustment to baseline value
                    baseline = series.mean()
                    calendar_forecast.append(baseline * adjustment)
                    
                forecast["calendar_forecast"] = calendar_forecast
        
        # Combine forecasts if multiple methods available
        if "acf_forecast" in forecast and "calendar_forecast" in forecast:
            combined_forecast = []
            for i in range(min(len(forecast["acf_forecast"]), len(forecast["calendar_forecast"]))):
                acf_weight = 0.6  # Weight for ACF forecast
                calendar_weight = 0.4  # Weight for calendar forecast
                combined_forecast.append(
                    acf_weight * forecast["acf_forecast"][i] + 
                    calendar_weight * forecast["calendar_forecast"][i]
                )
            forecast["combined_forecast"] = combined_forecast
                
        return {
            "forecast": forecast.get("combined_forecast", 
                       forecast.get("acf_forecast", 
                       forecast.get("calendar_forecast"))),
            "has_seasonality": results["has_seasonality"],
            "details": forecast
        }
