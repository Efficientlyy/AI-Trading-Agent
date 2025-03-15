"""Unit tests for enhanced price prediction v2 components."""

import unittest
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from src.ml.models.enhanced_price_prediction_v2 import (
    TechnicalFeatures,
    FeatureExtractor,
    ModelPredictor,
    FeatureVector
)
from src.ml.models.model_training import ModelTrainer, ModelValidator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class TestTechnicalFeatures(unittest.TestCase):
    """Test cases for technical feature calculations."""
    
    def setUp(self):
        """Set up test data."""
        self.prices = np.array([100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0], dtype=np.float64)
        self.technical = TechnicalFeatures()
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi = self.technical.calculate_rsi(self.prices, period=2)
        self.assertIsInstance(rsi, float)
        self.assertTrue(0 <= rsi <= 100)
    
    def test_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        short_prices = np.array([100.0], dtype=np.float64)
        rsi = self.technical.calculate_rsi(short_prices, period=2)
        self.assertEqual(rsi, 50.0)  # Should return neutral value
    
    def test_rsi_constant_prices(self):
        """Test RSI calculation with constant prices."""
        constant_prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        rsi = self.technical.calculate_rsi(constant_prices, period=2)
        self.assertEqual(rsi, 100.0)  # No downward movement
    
    def test_bb_position(self):
        """Test Bollinger Band position calculation."""
        position = self.technical.calculate_bb_position(self.prices, 102.0)
        self.assertIsInstance(position, float)
        self.assertTrue(0 <= position <= 1)
    
    def test_bb_position_insufficient_data(self):
        """Test BB position with insufficient data."""
        short_prices = np.array([100.0], dtype=np.float64)
        position = self.technical.calculate_bb_position(short_prices, 100.0)
        self.assertEqual(position, 0.5)  # Should return neutral value
    
    def test_bb_position_constant_prices(self):
        """Test BB position with constant prices."""
        constant_prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        position = self.technical.calculate_bb_position(constant_prices, 100.0)
        self.assertEqual(position, 0.5)  # Should return middle position
    
    def test_trend_strength(self):
        """Test trend strength calculation."""
        strength = self.technical.calculate_trend_strength(self.prices, period=3)
        self.assertIsInstance(strength, float)
        self.assertTrue(0 <= strength <= 1)
    
    def test_trend_strength_insufficient_data(self):
        """Test trend strength with insufficient data."""
        short_prices = np.array([100.0], dtype=np.float64)
        strength = self.technical.calculate_trend_strength(short_prices, period=3)
        self.assertEqual(strength, 0.0)  # Should return no trend
    
    def test_trend_strength_constant_prices(self):
        """Test trend strength with constant prices."""
        constant_prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        strength = self.technical.calculate_trend_strength(constant_prices, period=2)
        self.assertEqual(strength, 0.0)  # Should return no trend

class TestFeatureExtractor(unittest.TestCase):
    """Test cases for feature extraction."""
    
    def setUp(self):
        """Set up test data."""
        self.prices = np.array([100.0, 101.0, 99.0, 102.0, 103.0], dtype=np.float64)
        self.sentiment_data = {
            "social_sentiment": 0.5,
            "news_sentiment": 0.7,
            "order_flow_sentiment": 0.3,
            "fear_greed_index": 65.0
        }
        self.market_data = {
            "liquidity_score": 0.8,
            "volatility": 0.2,
            "correlation_score": 0.4
        }
        self.extractor = FeatureExtractor()
    
    def test_feature_extraction(self):
        """Test complete feature extraction."""
        features = self.extractor.extract_features(
            self.prices,
            self.sentiment_data,
            self.market_data
        )
        
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features["technical"]), 3)
        self.assertEqual(len(features["sentiment"]), 4)
        self.assertEqual(len(features["market"]), 3)
    
    def test_feature_extraction_missing_sentiment(self):
        """Test feature extraction with missing sentiment data."""
        incomplete_sentiment = {
            "social_sentiment": 0.5,
            "news_sentiment": 0.7
        }
        features = self.extractor.extract_features(
            self.prices,
            incomplete_sentiment,
            self.market_data
        )
        self.assertEqual(len(features["sentiment"]), 4)  # Should still return all features
    
    def test_feature_extraction_missing_market(self):
        """Test feature extraction with missing market data."""
        incomplete_market = {
            "liquidity_score": 0.8
        }
        features = self.extractor.extract_features(
            self.prices,
            self.sentiment_data,
            incomplete_market
        )
        self.assertEqual(len(features["market"]), 3)  # Should still return all features
    
    def test_feature_combination(self):
        """Test feature vector combination."""
        features = self.extractor.extract_features(
            self.prices,
            self.sentiment_data,
            self.market_data
        )
        combined = self.extractor.combine_features(features)
        
        self.assertIsInstance(combined, np.ndarray)
        self.assertEqual(len(combined), 10)  # 3 + 4 + 3 features
    
    def test_feature_combination_empty(self):
        """Test feature combination with empty features."""
        empty_features: FeatureVector = {
            "technical": np.zeros(3, dtype=np.float64),
            "sentiment": np.zeros(4, dtype=np.float64),
            "market": np.zeros(3, dtype=np.float64)
        }
        combined = self.extractor.combine_features(empty_features)
        self.assertEqual(len(combined), 10)
        self.assertTrue(np.all(combined == 0))
    
    def test_feature_extraction_invalid_prices(self):
        """Test feature extraction with invalid price data."""
        invalid_prices = np.array([], dtype=np.float64)
        features = self.extractor.extract_features(
            invalid_prices,
            self.sentiment_data,
            self.market_data
        )
        self.assertEqual(len(features["technical"]), 3)  # Should return zeros
        self.assertTrue(np.all(features["technical"] == 0))
    
    def test_feature_extraction_nan_prices(self):
        """Test feature extraction with NaN prices."""
        nan_prices = np.array([100.0, np.nan, 102.0], dtype=np.float64)
        features = self.extractor.extract_features(
            nan_prices,
            self.sentiment_data,
            self.market_data
        )
        self.assertEqual(len(features["technical"]), 3)
        self.assertFalse(np.any(np.isnan(features["technical"])))  # Should handle NaNs
    
    def test_feature_extraction_invalid_sentiment_values(self):
        """Test feature extraction with invalid sentiment values."""
        invalid_sentiment = {
            "social_sentiment": 1.5,  # Above valid range
            "news_sentiment": -0.5,   # Below valid range
            "order_flow_sentiment": np.nan,  # NaN
            "fear_greed_index": 150.0  # Above valid range
        }
        features = self.extractor.extract_features(
            self.prices,
            invalid_sentiment,
            self.market_data
        )
        # Check that values are clipped to valid ranges
        self.assertTrue(all(0 <= x <= 1 for x in features["sentiment"][:3]))
        self.assertTrue(0 <= features["sentiment"][3] <= 100)

class TestModelPredictor(unittest.TestCase):
    """Test cases for model prediction."""
    
    def setUp(self):
        """Set up test data and model."""
        self.features = np.random.rand(10).astype(np.float64)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.scaler = StandardScaler()
        
        # Train model on random data
        X = np.random.rand(100, 10).astype(np.float64)
        y = np.random.randint(0, 2, 100).astype(np.float64)
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        
        self.predictor = ModelPredictor(self.model, self.scaler)
    
    def test_prediction_generation(self):
        """Test prediction generation."""
        prediction = self.predictor.predict(self.features)
        
        self.assertIsInstance(prediction, dict)
        self.assertIn("prediction", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("direction", prediction)
        self.assertIn("timestamp", prediction)
        self.assertIn("features", prediction)
    
    def test_prediction_zero_features(self):
        """Test prediction with zero features."""
        zero_features = np.zeros(10, dtype=np.float64)
        prediction = self.predictor.predict(zero_features)
        self.assertIsInstance(prediction["prediction"], float)
        self.assertIsInstance(prediction["confidence"], float)
    
    def test_prediction_extreme_values(self):
        """Test prediction with extreme feature values."""
        extreme_features = np.array([1e6] * 10, dtype=np.float64)
        prediction = self.predictor.predict(extreme_features)
        self.assertIsInstance(prediction["prediction"], float)
        self.assertIsInstance(prediction["confidence"], float)
    
    def test_prediction_nan_features(self):
        """Test prediction with NaN features."""
        nan_features = np.array([np.nan] * 10, dtype=np.float64)
        prediction = self.predictor.predict(nan_features)
        self.assertEqual(prediction["direction"], 0)  # Should return neutral
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
    
    def test_prediction_inf_features(self):
        """Test prediction with infinite feature values."""
        inf_features = np.array([np.inf] * 10, dtype=np.float64)
        prediction = self.predictor.predict(inf_features)
        self.assertIsInstance(prediction["prediction"], float)
        self.assertTrue(np.isfinite(prediction["prediction"]))
    
    def test_prediction_wrong_shape(self):
        """Test prediction with wrong feature shape."""
        wrong_shape = np.random.rand(5).astype(np.float64)  # Wrong number of features
        with self.assertRaises(ValueError):
            self.predictor.predict(wrong_shape)

class TestModelTrainer(unittest.TestCase):
    """Test cases for model training."""
    
    def setUp(self):
        """Set up test data."""
        self.features = np.random.rand(100, 10).astype(np.float64)
        self.labels = np.random.randint(0, 2, 100).astype(np.float64)
        self.trainer = ModelTrainer(model_type="random_forest", n_splits=3)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.trainer.model, RandomForestClassifier)
    
    def test_model_creation_invalid_type(self):
        """Test model creation with invalid type."""
        with self.assertRaises(ValueError):
            ModelTrainer(model_type="invalid_model")
    
    def test_training_and_validation(self):
        """Test training and validation process."""
        metrics_list = self.trainer.train_and_validate(self.features, self.labels)
        
        self.assertEqual(len(metrics_list), 3)  # n_splits = 3
        for metrics in metrics_list:
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
    
    def test_training_with_sample_weights(self):
        """Test training with sample weights."""
        weights = np.ones(len(self.labels), dtype=np.float64)
        metrics_list = self.trainer.train_and_validate(
            self.features,
            self.labels,
            sample_weights=weights
        )
        self.assertEqual(len(metrics_list), 3)
    
    def test_training_imbalanced_data(self):
        """Test training with imbalanced data."""
        # Create imbalanced labels (90% class 0, 10% class 1)
        imbalanced_labels = np.zeros(100, dtype=np.float64)
        imbalanced_labels[:10] = 1.0
        
        metrics_list = self.trainer.train_and_validate(self.features, imbalanced_labels)
        self.assertEqual(len(metrics_list), 3)
    
    def test_training_with_nan_features(self):
        """Test training with NaN features."""
        nan_features = np.full((100, 10), np.nan, dtype=np.float64)
        with self.assertRaises(ValueError):
            self.trainer.train_and_validate(nan_features, self.labels)
    
    def test_training_mismatched_samples(self):
        """Test training with mismatched features and labels."""
        mismatched_labels = np.random.randint(0, 2, 50).astype(np.float64)  # Wrong length
        with self.assertRaises(ValueError):
            self.trainer.train_and_validate(self.features, mismatched_labels)
    
    def test_training_single_class(self):
        """Test training with single class data."""
        single_class_labels = np.zeros(100, dtype=np.float64)  # All zeros
        metrics_list = self.trainer.train_and_validate(self.features, single_class_labels)
        for metrics in metrics_list:
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertTrue(0 <= metrics["precision"] <= 1)

class TestModelValidator(unittest.TestCase):
    """Test cases for model validation."""
    
    def setUp(self):
        """Set up test data."""
        self.predictions = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
        self.returns = np.array([0.01, -0.01, 0.02, -0.015, 0.01], dtype=np.float64)
        self.validator = ModelValidator()
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        pf = self.validator.calculate_profit_factor(self.predictions, self.returns)
        self.assertIsInstance(pf, float)
        self.assertTrue(pf >= 0)
    
    def test_profit_factor_no_trades(self):
        """Test profit factor with no trades."""
        zero_preds = np.zeros_like(self.predictions)
        pf = self.validator.calculate_profit_factor(zero_preds, self.returns)
        self.assertEqual(pf, 0.0)
    
    def test_profit_factor_no_losses(self):
        """Test profit factor with no losing trades."""
        pos_returns = np.abs(self.returns)
        pf = self.validator.calculate_profit_factor(self.predictions, pos_returns)
        self.assertEqual(pf, 0.0)  # Should handle division by zero
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sr = self.validator.calculate_sharpe_ratio(self.predictions, self.returns)
        self.assertIsInstance(sr, float)
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        const_returns = np.ones_like(self.returns) * 0.01
        sr = self.validator.calculate_sharpe_ratio(self.predictions, const_returns)
        self.assertEqual(sr, 0.0)  # Should handle division by zero
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        mdd = self.validator.calculate_max_drawdown(self.predictions, self.returns)
        self.assertIsInstance(mdd, float)
        self.assertTrue(0 <= mdd <= 1)
    
    def test_max_drawdown_constant_equity(self):
        """Test maximum drawdown with constant equity curve."""
        const_returns = np.zeros_like(self.returns)
        mdd = self.validator.calculate_max_drawdown(self.predictions, const_returns)
        self.assertEqual(mdd, 0.0)
    
    def test_profit_factor_mismatched_lengths(self):
        """Test profit factor with mismatched array lengths."""
        short_returns = np.array([0.01, -0.01], dtype=np.float64)
        with self.assertRaises(ValueError):
            self.validator.calculate_profit_factor(self.predictions, short_returns)
    
    def test_sharpe_ratio_negative_rate(self):
        """Test Sharpe ratio with negative risk-free rate."""
        sr = self.validator.calculate_sharpe_ratio(
            self.predictions,
            self.returns,
            risk_free_rate=-0.01
        )
        self.assertIsInstance(sr, float)
    
    def test_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        single_value = np.array([0.01], dtype=np.float64)
        mdd = self.validator.calculate_max_drawdown(
            np.array([1.0], dtype=np.float64),
            single_value
        )
        self.assertEqual(mdd, 0.0)  # Should handle single value case

if __name__ == "__main__":
    unittest.main() 