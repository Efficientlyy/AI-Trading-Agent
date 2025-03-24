"""
Data Quality Factor

This module implements a confidence factor that evaluates prediction confidence
based on input data quality.
"""

from typing import Dict, Any, Optional, List, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.detection.enhanced_ensemble.confidence.factors.base import ConfidenceFactor


class DataQualityFactor(ConfidenceFactor):
    """
    Evaluates confidence based on input data quality.
    
    This factor assesses the quality of input data, including completeness,
    recency, and validity, to determine how reliable predictions might be.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data quality factor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._quality_metadata = {}
        
    def calculate(self, 
                detector_outputs: Dict[str, Dict[str, Any]],
                market_context: Dict[str, Any],
                historical_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate confidence based on data quality.
        
        Args:
            detector_outputs: Dictionary mapping detector names to their outputs
            market_context: Dictionary containing current market conditions
            historical_data: Optional historical market data
            
        Returns:
            Confidence factor score between 0.0 and 1.0
        """
        # Clear previous metadata
        self._quality_metadata = {}
        
        # Data quality assessment requires historical data
        if historical_data is None or historical_data.empty:
            self.logger.warning("No historical data provided for data quality assessment")
            return self.config.get('default_error_score', 0.5)
        
        # Calculate data quality metrics
        completeness_score = self._calculate_completeness(historical_data)
        recency_score = self._calculate_recency(historical_data, market_context)
        validity_score = self._calculate_validity(historical_data)
        
        # Store scores for metadata
        self._quality_metadata = {
            'completeness_score': completeness_score,
            'recency_score': recency_score,
            'validity_score': validity_score
        }
        
        # Calculate weighted average of quality metrics
        weights = self.config.get('quality_weights', {
            'completeness': 0.4,
            'recency': 0.4,
            'validity': 0.2
        })
        
        final_score = (
            weights.get('completeness', 0.4) * completeness_score +
            weights.get('recency', 0.4) * recency_score +
            weights.get('validity', 0.2) * validity_score
        )
        
        self._quality_metadata["final_quality_score"] = final_score
        
        return final_score
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """
        Calculate data completeness score.
        
        Args:
            data: Historical market data
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        # Calculate percentage of non-null values
        if len(data) == 0:
            return 0.0
            
        # Get required columns based on configuration
        required_columns = self.config.get('required_columns', [])
        
        # If no required columns specified, use all columns
        if not required_columns:
            required_columns = data.columns.tolist()
            
        # Calculate completeness for required columns
        completeness_scores = {}
        for column in required_columns:
            if column in data.columns:
                non_null_ratio = data[column].notnull().mean()
                completeness_scores[column] = non_null_ratio
            else:
                # Column is missing entirely
                completeness_scores[column] = 0.0
                
        # Store column-wise completeness for metadata
        self._quality_metadata["column_completeness"] = completeness_scores
                
        # Overall completeness is the average of column completeness
        if not completeness_scores:
            return 0.0
            
        return sum(completeness_scores.values()) / len(completeness_scores)
    
    def _calculate_recency(self, 
                         data: pd.DataFrame,
                         market_context: Dict[str, Any]) -> float:
        """
        Calculate data recency score.
        
        Args:
            data: Historical market data
            market_context: Market context containing timestamp
            
        Returns:
            Recency score between 0.0 and 1.0
        """
        # Extract current timestamp from market context
        current_timestamp = market_context.get('timestamp')
        
        # If no timestamp in context, use current time
        if current_timestamp is None:
            current_timestamp = datetime.now()
            
        # Convert to datetime if string
        if isinstance(current_timestamp, str):
            try:
                current_timestamp = pd.to_datetime(current_timestamp)
            except:
                self.logger.warning("Could not parse timestamp from market context")
                return 0.5
        
        # Check if data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to convert the first column to datetime if it looks like a date
            date_columns = [col for col in data.columns 
                           if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                try:
                    data_timestamps = pd.to_datetime(data[date_columns[0]])
                except:
                    self.logger.warning("Could not find or parse date/time column")
                    return 0.5
            else:
                self.logger.warning("No datetime index or date/time column found")
                return 0.5
        else:
            data_timestamps = data.index
            
        # Calculate how recent the data is
        latest_timestamp = data_timestamps.max()
        
        # Calculate the time difference in hours
        try:
            time_diff_hours = (current_timestamp - latest_timestamp).total_seconds() / 3600
        except:
            self.logger.warning("Error calculating time difference")
            return 0.5
            
        # Normalize based on expected data frequency
        expected_update_hours = self.config.get('expected_update_hours', 24)
        
        # Store time difference for metadata
        self._quality_metadata["time_diff_hours"] = time_diff_hours
        self._quality_metadata["expected_update_hours"] = expected_update_hours
        
        # Calculate recency score (1.0 if up-to-date, decreasing as data ages)
        if time_diff_hours <= 0:
            # Data is more recent than current timestamp (should be rare)
            return 1.0
        elif time_diff_hours >= expected_update_hours * 3:
            # Data is very old (3x the expected update frequency)
            return 0.1
        else:
            # Linear scaling between 1.0 and 0.1
            recency_score = 1.0 - (time_diff_hours / (expected_update_hours * 3)) * 0.9
            return max(0.1, recency_score)
    
    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """
        Calculate data validity score based on outliers and invalid values.
        
        Args:
            data: Historical market data
            
        Returns:
            Validity score between 0.0 and 1.0
        """
        # Use only numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            self.logger.warning("No numeric columns found for validity check")
            return 0.5
            
        # Calculate percentage of outliers and invalid values
        validity_scores = {}
        
        for column in numeric_columns:
            col_data = data[column]
            
            # Skip columns with all NaN
            if col_data.isna().all():
                continue
                
            # Check for infinities and NaNs
            invalid_mask = ~np.isfinite(col_data)
            invalid_ratio = invalid_mask.mean()
            
            # Check for outliers using IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_ratio = outlier_mask.mean()
            
            # Combined validity score for this column
            column_validity = 1.0 - (invalid_ratio + min(outlier_ratio, 0.3))
            validity_scores[column] = max(0.0, column_validity)
            
        # Store column-wise validity for metadata
        self._quality_metadata["column_validity"] = validity_scores
            
        # Overall validity is the average of column validity scores
        if not validity_scores:
            return 0.5
            
        return sum(validity_scores.values()) / len(validity_scores)
            
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get additional metadata about this factor calculation.
        
        Returns:
            Dictionary of metadata that can be used for analysis and debugging
        """
        return {
            'quality_metrics': self._quality_metadata
        }
