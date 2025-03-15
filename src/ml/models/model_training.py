"""Model training and validation components for enhanced price prediction."""

from typing import Dict, List, Optional, Tuple, Union, TypedDict
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from datetime import datetime

class TrainingMetrics(TypedDict):
    """Type definition for training metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_samples: int
    validation_samples: int
    training_time: float
    timestamp: float

class ModelTrainer:
    """Model training with cross-validation and metrics tracking."""
    
    def __init__(
        self,
        model_type: str = "random_forest",
        n_splits: int = 5,
        test_size: float = 0.2
    ):
        self.model_type = model_type
        self.n_splits = n_splits
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.model = self._create_model()
        
    def _create_model(self) -> Union[RandomForestClassifier, GradientBoostingClassifier]:
        """Create model instance based on type."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train_and_validate(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None
    ) -> List[TrainingMetrics]:
        """Train and validate model using time series cross-validation."""
        metrics_list = []
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=int(len(features) * self.test_size))
        
        for train_idx, val_idx in tscv.split(features):
            start_time = datetime.now()
            
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            if sample_weights is not None:
                train_weights = sample_weights[train_idx]
                self.model.fit(X_train_scaled, y_train, sample_weight=train_weights)
            else:
                self.model.fit(X_train_scaled, y_train)
            
            # Generate predictions
            y_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            metrics: TrainingMetrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
                "training_time": float((datetime.now() - start_time).total_seconds()),
                "timestamp": float(datetime.now().timestamp())
            }
            
            metrics_list.append(metrics)
        
        return metrics_list

class ModelValidator:
    """Model validation and performance analysis."""
    
    @staticmethod
    def calculate_profit_factor(
        predictions: NDArray[np.float64],
        actual_returns: NDArray[np.float64]
    ) -> float:
        """Calculate profit factor from predictions."""
        try:
            # Filter trades based on predictions
            predicted_trades = predictions * actual_returns
            
            # Calculate wins and losses
            wins = np.sum(predicted_trades[predicted_trades > 0])
            losses = abs(np.sum(predicted_trades[predicted_trades < 0]))
            
            return float(wins / losses) if losses != 0 else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_sharpe_ratio(
        predictions: NDArray[np.float64],
        actual_returns: NDArray[np.float64],
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio for the strategy."""
        try:
            # Calculate strategy returns
            strategy_returns = predictions * actual_returns
            
            # Annualize metrics (assuming daily data)
            annual_return = np.mean(strategy_returns) * 252
            annual_volatility = np.std(strategy_returns) * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0.0
                
            return float((annual_return - risk_free_rate) / annual_volatility)
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(
        predictions: NDArray[np.float64],
        actual_returns: NDArray[np.float64]
    ) -> float:
        """Calculate maximum drawdown of the strategy."""
        try:
            # Calculate strategy returns
            strategy_returns = predictions * actual_returns
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + strategy_returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            return float(abs(np.min(drawdowns)))
        except Exception:
            return 0.0 