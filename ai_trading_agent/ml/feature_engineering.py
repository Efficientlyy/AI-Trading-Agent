"""
Automated Feature Engineering Module for the AI Trading Agent

This module implements automated feature engineering capabilities including:
- Feature importance ranking
- Automatic feature selection
- Dynamic feature creation based on market regimes
- Feature correlation analysis

The module is designed to work in tandem with the reinforcement learning system
to continuously adapt the feature set based on market conditions and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings

from ai_trading_agent.utils.logging import get_logger
from ai_trading_agent.agent.market_regime import MarketRegimeClassifier

# Configure logger
logger = get_logger(__name__)

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class FeatureEngineer:
    """
    Automated feature engineering system for trading data.
    Provides feature importance ranking, feature selection, and
    dynamic feature creation based on market regimes.
    """
    
    def __init__(
        self,
        base_features: List[str],
        target_column: str = 'returns',
        regime_classifier: Optional[MarketRegimeClassifier] = None,
        max_features: int = 20,
        importance_method: str = 'random_forest',
        feature_creation_methods: Optional[List[str]] = None,
        window_sizes: Optional[List[int]] = None,
        selection_threshold: float = 0.01
    ):
        """
        Initialize the Feature Engineer.
        
        Args:
            base_features: List of base feature names to start with
            target_column: Name of the target column for importance calculation
            regime_classifier: Optional classifier for market regime detection
            max_features: Maximum number of features to select
            importance_method: Method to use for feature importance ranking
                               ('random_forest', 'gradient_boosting', 'mutual_info')
            feature_creation_methods: Methods to use for creating new features
            window_sizes: List of window sizes for feature creation
            selection_threshold: Minimum importance score for feature selection
        """
        self.base_features = base_features
        self.target_column = target_column
        self.regime_classifier = regime_classifier
        self.max_features = max_features
        self.importance_method = importance_method
        self.feature_creation_methods = feature_creation_methods or [
            'ma', 'std', 'rsi', 'momentum', 'ratio', 'diff', 'log_return'
        ]
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
        self.selection_threshold = selection_threshold
        
        # Feature tracking
        self.feature_importances = {}
        self.selected_features = []
        self.feature_stats = {}
        self.created_features = set()
        self.regime_specific_features = {}
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Feature Engineer initialized with {len(self.base_features)} base features")
    
    def create_features(self, data: pd.DataFrame, regime: Optional[str] = None) -> pd.DataFrame:
        """
        Create new features based on base features and market regime.
        
        Args:
            data: DataFrame with market data
            regime: Current market regime (if available)
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        created = set()
        
        # Apply regime-specific feature creation if regime is provided
        if regime and regime in self.regime_specific_features:
            feature_methods = self.regime_specific_features[regime]
        else:
            # Default to all feature creation methods
            feature_methods = self.feature_creation_methods
        
        # Apply feature creation methods
        for method in feature_methods:
            if method == 'ma':
                # Moving averages
                for feature in self.base_features:
                    if feature in df.columns:
                        for window in self.window_sizes:
                            col_name = f'{feature}_ma_{window}'
                            df[col_name] = df[feature].rolling(window=window).mean()
                            created.add(col_name)
            
            elif method == 'std':
                # Standard deviation (volatility)
                for feature in self.base_features:
                    if feature in df.columns:
                        for window in self.window_sizes:
                            col_name = f'{feature}_std_{window}'
                            df[col_name] = df[feature].rolling(window=window).std()
                            created.add(col_name)
            
            elif method == 'rsi':
                # Relative Strength Index
                for feature in self.base_features:
                    if feature in df.columns:
                        for window in self.window_sizes:
                            col_name = f'{feature}_rsi_{window}'
                            delta = df[feature].diff()
                            gain = delta.where(delta > 0, 0)
                            loss = -delta.where(delta < 0, 0)
                            avg_gain = gain.rolling(window=window).mean()
                            avg_loss = loss.rolling(window=window).mean()
                            rs = avg_gain / avg_loss.replace(0, np.nan)
                            df[col_name] = 100 - (100 / (1 + rs))
                            df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)
                            created.add(col_name)
            
            elif method == 'momentum':
                # Momentum
                for feature in self.base_features:
                    if feature in df.columns:
                        for window in self.window_sizes:
                            col_name = f'{feature}_momentum_{window}'
                            df[col_name] = df[feature] - df[feature].shift(window)
                            created.add(col_name)
            
            elif method == 'ratio':
                # Ratios between features
                if len(self.base_features) >= 2:
                    for i, feat1 in enumerate(self.base_features):
                        if feat1 not in df.columns:
                            continue
                        for feat2 in self.base_features[i+1:]:
                            if feat2 not in df.columns:
                                continue
                            col_name = f'{feat1}_div_{feat2}'
                            df[col_name] = df[feat1] / df[feat2].replace(0, np.nan)
                            df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)
                            created.add(col_name)
            
            elif method == 'diff':
                # Differences and percentage changes
                for feature in self.base_features:
                    if feature in df.columns:
                        # First difference
                        col_name = f'{feature}_diff'
                        df[col_name] = df[feature].diff()
                        created.add(col_name)
                        
                        # Percentage change
                        col_name = f'{feature}_pct_change'
                        df[col_name] = df[feature].pct_change()
                        created.add(col_name)
            
            elif method == 'log_return':
                # Log returns
                for feature in self.base_features:
                    if feature in df.columns:
                        col_name = f'{feature}_log_return'
                        df[col_name] = np.log(df[feature] / df[feature].shift(1))
                        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)
                        created.add(col_name)
        
        # Update the created features set
        self.created_features.update(created)
        
        # Fill NaN values with 0 for created features
        df = df.fillna(0)
        
        logger.info(f"Created {len(created)} new features, total features: {len(df.columns)}")
        return df
    
    def rank_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Rank features by importance with respect to the target variable.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.target_column not in data.columns:
            logger.error(f"Target column '{self.target_column}' not found in data")
            return {}
        
        # Drop rows with NaN values
        df = data.dropna()
        
        if len(df) == 0:
            logger.error("No valid data after dropping NaN values")
            return {}
        
        # Select features (exclude target and non-numeric columns)
        features = [col for col in df.columns if col != self.target_column 
                    and np.issubdtype(df[col].dtype, np.number)]
        
        if not features:
            logger.error("No numeric feature columns found")
            return {}
        
        X = df[features]
        y = df[self.target_column]
        
        importances = {}
        
        try:
            if self.importance_method == 'random_forest':
                # Random Forest for feature importance
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                importance_scores = model.feature_importances_
                
            elif self.importance_method == 'gradient_boosting':
                # Gradient Boosting for feature importance
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                importance_scores = model.feature_importances_
                
            elif self.importance_method == 'mutual_info':
                # Mutual Information for feature importance
                importance_scores = mutual_info_regression(X, y, random_state=42)
            
            else:
                logger.error(f"Unknown importance method: {self.importance_method}")
                return {}
            
            # Create dictionary of feature importances
            importances = {feature: float(score) for feature, score in zip(features, importance_scores)}
            
            # Sort by importance (descending)
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            
            # Update feature importance tracking
            self.feature_importances = importances
            
            logger.info(f"Ranked {len(importances)} features by importance")
            
        except Exception as e:
            logger.error(f"Error ranking features: {str(e)}")
            return {}
        
        return importances
    
    def select_features(self, importances: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Select top features based on importance scores.
        
        Args:
            importances: Optional dictionary of feature importance scores
                        (uses self.feature_importances if not provided)
            
        Returns:
            List of selected feature names
        """
        if importances is None:
            importances = self.feature_importances
        
        if not importances:
            logger.warning("No feature importances available for selection")
            return self.base_features  # Fall back to base features
        
        # Filter features by threshold
        filtered_features = {f: score for f, score in importances.items() 
                            if score >= self.selection_threshold}
        
        # Take top K features
        top_features = list(filtered_features.keys())[:self.max_features]
        
        # Update selected features
        self.selected_features = top_features
        
        logger.info(f"Selected {len(top_features)} features")
        return top_features
    
    def adapt_features_to_regime(self, regime: str, performance: float) -> None:
        """
        Adapt feature creation based on performance in a specific market regime.
        
        Args:
            regime: Current market regime
            performance: Performance metric (e.g., Sharpe ratio) for this regime
        """
        if regime not in self.regime_specific_features:
            # Initialize with all methods for new regime
            self.regime_specific_features[regime] = self.feature_creation_methods
        
        # Track performance
        self.performance_history.append({
            'regime': regime,
            'performance': performance,
            'features': self.regime_specific_features[regime]
        })
        
        # Need at least a few data points before adapting
        if len(self.performance_history) < 5:
            return
        
        # Filter history for this regime
        regime_history = [entry for entry in self.performance_history if entry['regime'] == regime]
        
        if len(regime_history) < 3:
            return
        
        # Get best performing feature set for this regime
        best_entry = max(regime_history, key=lambda x: x['performance'])
        
        # Compare current performance to best
        current_performance = performance
        best_performance = best_entry['performance']
        
        # If performance declined significantly, experiment with feature methods
        if current_performance < best_performance * 0.8:
            # Try removing a random method or adding a new one
            current_methods = set(self.regime_specific_features[regime])
            all_methods = set(self.feature_creation_methods)
            
            if len(current_methods) > 1 and np.random.random() > 0.5:
                # Remove a method
                method_to_remove = np.random.choice(list(current_methods))
                new_methods = current_methods - {method_to_remove}
                logger.info(f"Removing feature method '{method_to_remove}' for regime '{regime}'")
            else:
                # Add a method
                available_to_add = all_methods - current_methods
                if available_to_add:
                    method_to_add = np.random.choice(list(available_to_add))
                    new_methods = current_methods | {method_to_add}
                    logger.info(f"Adding feature method '{method_to_add}' for regime '{regime}'")
                else:
                    # No methods to add, keep current
                    new_methods = current_methods
            
            self.regime_specific_features[regime] = list(new_methods)
        elif current_performance > best_performance * 1.1:
            # If performance improved significantly, save this feature set
            logger.info(f"New best feature set for regime '{regime}' with performance {current_performance}")
    
    def dimensionality_reduction(self, data: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Reduce feature dimensionality using PCA.
        
        Args:
            data: DataFrame with features
            n_components: Number of components to keep
            
        Returns:
            DataFrame with reduced features
        """
        # Select only numeric columns
        numeric_cols = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]
        
        if not numeric_cols:
            logger.error("No numeric feature columns found for dimensionality reduction")
            return data
        
        # Create a pipeline with scaling and PCA
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=min(n_components, len(numeric_cols))))
        ])
        
        try:
            # Apply the pipeline
            transformed = pipeline.fit_transform(data[numeric_cols])
            
            # Create a new DataFrame with the PCA components
            pca_df = pd.DataFrame(
                transformed, 
                index=data.index, 
                columns=[f'pca_{i+1}' for i in range(transformed.shape[1])]
            )
            
            # Get the explained variance ratio
            pca = pipeline.named_steps['pca']
            explained_variance = pca.explained_variance_ratio_
            
            logger.info(f"PCA reduced {len(numeric_cols)} features to {transformed.shape[1]} components")
            logger.info(f"Explained variance: {sum(explained_variance):.2f}")
            
            return pca_df
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {str(e)}")
            return data
    
    def analyze_feature_correlations(self, data: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze correlations between features to identify redundancy.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Dictionary mapping each feature to a list of highly correlated features
        """
        # Select only numeric columns
        numeric_cols = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]
        
        if not numeric_cols:
            logger.error("No numeric feature columns found for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Identify high correlations for each feature
        high_correlations = {}
        
        for feature in numeric_cols:
            # Get correlations for this feature, sorted by absolute value
            correlations = [(other_feat, corr) 
                           for other_feat, corr in corr_matrix[feature].items()
                           if other_feat != feature and abs(corr) > 0.7]
            
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if correlations:
                high_correlations[feature] = correlations
        
        # Store feature stats
        self.feature_stats['high_correlations'] = high_correlations
        
        logger.info(f"Found {len(high_correlations)} features with high correlations")
        return high_correlations
    
    def get_optimal_feature_set(self, 
                               data: pd.DataFrame, 
                               regime: Optional[str] = None) -> pd.DataFrame:
        """
        Get the optimal feature set for the current data and market regime.
        This is the main method to call for end-to-end feature engineering.
        
        Args:
            data: DataFrame with original data
            regime: Current market regime (if available)
            
        Returns:
            DataFrame with optimal feature set
        """
        # Create features based on regime
        df_with_features = self.create_features(data, regime)
        
        # Rank features by importance
        importances = self.rank_features(df_with_features)
        
        # Select top features
        selected = self.select_features(importances)
        
        # Include target column if present
        if self.target_column in df_with_features.columns:
            selected.append(self.target_column)
        
        # Get the final feature set
        result = df_with_features[selected]
        
        logger.info(f"Generated optimal feature set with {len(selected)} features")
        return result


# Factory function to create a feature engineer
def create_feature_engineer(
    config: Dict[str, Any],
    market_regime_classifier: Optional[MarketRegimeClassifier] = None
) -> FeatureEngineer:
    """
    Create a Feature Engineer with the specified configuration.
    
    Args:
        config: Configuration dictionary
        market_regime_classifier: Optional market regime classifier
        
    Returns:
        Configured FeatureEngineer
    """
    # Extract configuration with defaults
    base_features = config.get('base_features', [
        'open', 'high', 'low', 'close', 'volume'
    ])
    
    target_column = config.get('target_column', 'returns')
    
    max_features = config.get('max_features', 20)
    
    importance_method = config.get('importance_method', 'random_forest')
    
    feature_creation_methods = config.get('feature_creation_methods', [
        'ma', 'std', 'rsi', 'momentum', 'ratio', 'diff', 'log_return'
    ])
    
    window_sizes = config.get('window_sizes', [5, 10, 20, 50, 100])
    
    selection_threshold = config.get('selection_threshold', 0.01)
    
    # Create and return feature engineer
    feature_engineer = FeatureEngineer(
        base_features=base_features,
        target_column=target_column,
        regime_classifier=market_regime_classifier,
        max_features=max_features,
        importance_method=importance_method,
        feature_creation_methods=feature_creation_methods,
        window_sizes=window_sizes,
        selection_threshold=selection_threshold
    )
    
    return feature_engineer
