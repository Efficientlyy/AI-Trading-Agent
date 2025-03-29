"""Enhanced adaptive learning module for sentiment analysis.

This module provides advanced functionality for adaptive learning in sentiment analysis,
including sophisticated weight adjustment algorithms and performance visualization tools.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
import os

class EnhancedAdaptiveWeights:
    """Enhanced adaptive weights system with advanced learning capabilities."""
    
    def __init__(self, 
                 learning_rate: float = 0.02,
                 decay_factor: float = 0.9,
                 performance_window: int = 60,
                 min_samples: int = 15,
                 visualization_dir: str = "/tmp/sentiment_visualization"):
        """Initialize the enhanced adaptive weights system.
        
        Args:
            learning_rate: Rate of weight adjustment
            decay_factor: Factor for time-based decay of old performance data
            performance_window: Days to keep performance data
            min_samples: Minimum samples needed before adjustment
            visualization_dir: Directory for saving visualization outputs
        """
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.performance_window = performance_window
        self.min_samples = min_samples
        self.visualization_dir = visualization_dir
        
        # Ensure visualization directory exists
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Default source weights
        self.source_weights = {
            "social_media": 0.25,
            "news": 0.25,
            "market_sentiment": 0.3,
            "onchain": 0.2
        }
        
        # Performance history by source and symbol
        self.source_performance: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            "social_media": {},
            "news": {},
            "market_sentiment": {},
            "onchain": {}
        }
        
        # Symbol-specific weights
        self.symbol_weights: Dict[str, Dict[str, float]] = {}
        
        # Market condition weights
        self.market_condition_weights: Dict[str, Dict[str, float]] = {
            "bullish": self.source_weights.copy(),
            "bearish": self.source_weights.copy(),
            "neutral": self.source_weights.copy(),
            "volatile": self.source_weights.copy()
        }
        
        # Weight adjustment history
        self.weight_history: Dict[str, List[Dict[str, Any]]] = {
            "global": [],
            "by_symbol": {},
            "by_market_condition": {}
        }
        
        # Last update timestamp
        self.last_update = datetime.utcnow()
        
        # Performance metrics
        self.performance_metrics = {
            "global_accuracy": 0.0,
            "by_source": {},
            "by_symbol": {},
            "by_market_condition": {}
        }
        
    def record_performance(self, 
                          source: str, 
                          symbol: str,
                          prediction: float,
                          actual_outcome: float,
                          timestamp: datetime,
                          market_condition: str = "neutral",
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the performance of a sentiment source prediction.
        
        Args:
            source: The sentiment source
            symbol: The trading pair symbol
            prediction: The predicted sentiment value (0-1)
            actual_outcome: The actual market movement normalized to 0-1
            timestamp: When the prediction was made
            market_condition: Market condition (bullish, bearish, neutral, volatile)
            metadata: Additional metadata about the prediction
        """
        if source not in self.source_performance:
            self.source_performance[source] = {}
            
        if symbol not in self.source_performance[source]:
            self.source_performance[source][symbol] = []
            
        # Calculate accuracy (inverted mean squared error)
        error = (prediction - actual_outcome) ** 2
        accuracy = max(0, 1 - error)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        # Record performance
        self.source_performance[source][symbol].append({
            "prediction": prediction,
            "actual": actual_outcome,
            "accuracy": accuracy,
            "timestamp": timestamp,
            "market_condition": market_condition,
            "metadata": metadata
        })
        
        # Clean up old performance data
        cutoff = datetime.utcnow() - timedelta(days=self.performance_window)
        self.source_performance[source][symbol] = [
            p for p in self.source_performance[source][symbol]
            if p["timestamp"] > cutoff
        ]
        
        # Update performance metrics
        self._update_performance_metrics()
        
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on recorded data."""
        # Reset metrics
        self.performance_metrics = {
            "global_accuracy": 0.0,
            "by_source": {},
            "by_symbol": {},
            "by_market_condition": {}
        }
        
        # Collect all performance data
        all_performances = []
        for source, symbols in self.source_performance.items():
            source_performances = []
            
            # Initialize source metrics
            self.performance_metrics["by_source"][source] = {
                "accuracy": 0.0,
                "sample_count": 0,
                "by_market_condition": {
                    "bullish": {"accuracy": 0.0, "count": 0},
                    "bearish": {"accuracy": 0.0, "count": 0},
                    "neutral": {"accuracy": 0.0, "count": 0},
                    "volatile": {"accuracy": 0.0, "count": 0}
                }
            }
            
            for symbol, performances in symbols.items():
                # Initialize symbol metrics if needed
                if symbol not in self.performance_metrics["by_symbol"]:
                    self.performance_metrics["by_symbol"][symbol] = {
                        "accuracy": 0.0,
                        "sample_count": 0,
                        "by_source": {}
                    }
                
                # Initialize symbol-source metrics if needed
                if source not in self.performance_metrics["by_symbol"][symbol]["by_source"]:
                    self.performance_metrics["by_symbol"][symbol]["by_source"][source] = {
                        "accuracy": 0.0,
                        "sample_count": 0
                    }
                
                # Add to collections
                source_performances.extend(performances)
                all_performances.extend(performances)
                
                # Update symbol-source metrics
                if performances:
                    avg_accuracy = np.mean([p["accuracy"] for p in performances])
                    self.performance_metrics["by_symbol"][symbol]["by_source"][source] = {
                        "accuracy": float(avg_accuracy),
                        "sample_count": len(performances)
                    }
                    
                    # Update symbol metrics
                    self.performance_metrics["by_symbol"][symbol]["sample_count"] += len(performances)
                    total_accuracy = self.performance_metrics["by_symbol"][symbol]["accuracy"] * (
                        self.performance_metrics["by_symbol"][symbol]["sample_count"] - len(performances)
                    )
                    total_accuracy += avg_accuracy * len(performances)
                    self.performance_metrics["by_symbol"][symbol]["accuracy"] = total_accuracy / self.performance_metrics["by_symbol"][symbol]["sample_count"]
            
            # Update source metrics
            if source_performances:
                avg_accuracy = np.mean([p["accuracy"] for p in source_performances])
                self.performance_metrics["by_source"][source]["accuracy"] = float(avg_accuracy)
                self.performance_metrics["by_source"][source]["sample_count"] = len(source_performances)
                
                # Update by market condition
                for condition in ["bullish", "bearish", "neutral", "volatile"]:
                    condition_performances = ["p for p in source_performances if p["market_condition""] = = condition]
                    if condition_performances:
                        condition_accuracy = np.mean([p["accuracy"] for p in condition_performances])
                        self.performance_metrics["by_source"][source]["by_market_condition"][condition] = {
                            "accuracy": float(condition_accuracy),
                            "count": len(condition_performances)
                        }
        
        # Update global accuracy
        if all_performances:
            self.performance_metrics["global_accuracy"] = float(np.mean([p["accuracy"] for p in all_performances]))
            
        # Update market condition metrics
        for condition in ["bullish", "bearish", "neutral", "volatile"]:
            condition_performances = ["p for p in all_performances if p["market_condition""] = = condition]
            if condition_performances:
                condition_accuracy = np.mean([p["accuracy"] for p in condition_performances])
                if "by_market_condition" not in self.performance_metrics:
                    self.performance_metrics["by_market_condition"] = {}
                self.performance_metrics["by_market_condition"][condition] = {
                    "accuracy": float(condition_accuracy),
                    "count": len(condition_performances)
                }
        
    def update_weights(self, 
                      symbol: Optional[str] = None,
                      market_condition: Optional[str] = None) -> Dict[str, float]:
        """Update weights based on recent performance.
        
        Args:
            symbol: Optional symbol for symbol-specific weights
            market_condition: Optional market condition for condition-specific weights
            
        Returns:
            The updated source weights
        """
        # Update global weights
        self._update_global_weights()
        
        # Update symbol-specific weights if requested
        if symbol:
            self._update_symbol_weights(symbol)
            
        # Update market condition weights if requested
        if market_condition:
            self._update_market_condition_weights(market_condition)
            
        # Return appropriate weights
        if symbol and symbol in self.symbol_weights:
            return self.symbol_weights[symbol]
        elif market_condition and market_condition in self.market_condition_weights:
            return self.market_condition_weights[market_condition]
        else:
            return self.source_weights
            
    def _update_global_weights(self) -> None:
        """Update global weights based on overall performance."""
        # Check if we have enough data
        has_enough_data = all(
            sum(len(perfs) for perfs in symbols.values()) >= self.min_samples
            for source, symbols in self.source_performance.items()
            if source in self.source_weights
        )
        
        if not has_enough_data:
            return
            
        # Calculate performance-based adjustments
        adjustments = {}
        total_adjustment = 0
        
        for source in self.source_weights.keys():
            if source not in self.source_performance:
                continue
                
            # Collect all performances for this source
            all_performances = []
            for symbol_perfs in self.source_performance[source].values():
                all_performances.extend(symbol_perfs)
                
            if not all_performances:
                continue
                
            # Calculate recent average accuracy with time decay
            now = datetime.utcnow()
            weighted_sum = 0
            weight_sum = 0
            
            for perf in all_performances:
                # More recent performances get higher weight
                days_old = (now - perf["timestamp"]).total_seconds() / 86400
                time_weight = self.decay_factor ** days_old
                
                weighted_sum += perf["accuracy"] * time_weight
                weight_sum += time_weight
                
            if weight_sum > 0:
                avg_accuracy = weighted_sum / weight_sum
            else:
                avg_accuracy = 0.5
                
            # Calculate adjustment (positive if accuracy > 0.5, negative otherwise)
            current_weight = self.source_weights.get(source, 0.25)
            target_weight = current_weight * (1 + (avg_accuracy - 0.5) * self.learning_rate)
            
            # Ensure weight stays in reasonable range
            target_weight = max(0.05, min(0.5, target_weight))
            
            adjustments[source] = target_weight - current_weight
            total_adjustment += abs(adjustments[source])
            
        # Apply adjustments while maintaining sum of weights = 1
        if total_adjustment > 0:
            # Apply raw adjustments first
            old_weights = self.source_weights.copy()
            for source in self.source_weights:
                if source in adjustments:
                    self.source_weights[source] += adjustments[source]
                    
            # Normalize weights to sum to 1
            weight_sum = sum(self.source_weights.values())
            for source in self.source_weights:
                self.source_weights[source] /= weight_sum
                
            # Record weight adjustment
            self.weight_history["global"].append({
                "timestamp": datetime.utcnow(),
                "old_weights": old_weights,
                "new_weights": self.source_weights.copy(),
                "adjustments": adjustments
            })
                
        # Update timestamp
        self.last_update = datetime.utcnow()
        
    def _update_symbol_weights(self, symbol: str) -> None:
        """Update weights for a specific symbol.
        
        Args:
            symbol: The trading pair symbol
        """
        # Check if we have data for this symbol
        has_data = all(
            symbol in self.source_performance[source] and 
            len(self.source_performance[source][symbol]) >= self.min_samples
            for source in self.source_weights.keys()
            if source in self.source_performance
        )
        
        if not has_data:
            # Use global weights if not enough symbol-specific data
            if symbol not in self.symbol_weights:
                self.symbol_weights[symbol] = self.source_weights.copy()
            return
            
        # Initialize symbol weights if needed
        if symbol not in self.symbol_weights:
            self.symbol_weights[symbol] = self.source_weights.copy()
            
        # Calculate performance-based adjustments
        adjustments = {}
        total_adjustment = 0
        
        for source in self.source_weights.keys():
            if (source not in self.source_performance or 
                symbol not in self.source_performance[source] or
                not self.source_performance[source][symbol]):
                continue
                
            # Get performances for this source and symbol
            performances = self.source_performance[source][symbol]
            
            # Calculate recent average accuracy with time decay
            now = datetime.utcnow()
            weighted_sum = 0
            weight_sum = 0
            
            for perf in performances:
                # More recent performances get higher weight
                days_old = (now - perf["timestamp"]).total_seconds() / 86400
                time_weight = self.decay_factor ** days_old
                
                weighted_sum += perf["accuracy"] * time_weight
                weight_sum += time_weight
                
            if weight_sum > 0:
                avg_accuracy = weighted_sum / weight_sum
            else:
                avg_accuracy = 0.5
                
            # Calculate adjustment (positive if accuracy > 0.5, negative otherwise)
            current_weight = self.symbol_weights[symbol].get(source, 0.25)
            target_weight = current_weight * (1 + (avg_accuracy - 0.5) * self.learning_rate)
            
            # Ensure weight stays in reasonable range
            target_weight = max(0.05, min(0.5, target_weight))
            
            adjustments[source] = target_weight - current_weight
            total_adjustment += abs(adjustments[source])
            
        # Apply adjustments while maintaining sum of weights = 1
        if total_adjustment > 0:
            # Apply raw adjustments first
            old_weights = self.symbol_weights[symbol].copy()
            for source in self.symbol_weights[symbol]:
                if source in adjustments:
                    self.symbol_weights[symbol][source] += adjustments[source]
                    
            # Normalize weights to sum to 1
            weight_sum = sum(self.symbol_weights[symbol].values())
            for source in self.symbol_weights[symbol]:
                self.symbol_weights[symbol][source] /= weight_sum
                
            # Record weight adjustment
            if symbol not in self.weight_history["by_symbol"]:
                self.weight_history["by_symbol"][symbol] = []
                
            self.weight_history["by_symbol"][symbol].append({
                "timestamp": datetime.utcnow(),
                "old_weights": old_weights,
                "new_weights": self.symbol_weights[symbol].copy(),
                "adjustments": adjustments
            })
        
    def _update_market_condition_weights(self, market_condition: str) -> None:
        """Update weights for a specific market condition.
        
        Args:
            market_condition: The market condition (bullish, bearish, neutral, volatile)
        """
        if market_condition not in ["bullish", "bearish", "neutral", "volatile"]:
            return
            
        # Check if we have enough data for this market condition
        condition_performances = {}
        
        for source in self.source_weights.keys():
            if source not in self.source_performance:
                continue
                
            # Collect all performances for this source and condition
            source_condition_perfs = []
            for symbol_perfs in self.source_performance[source].values():
                source_condition_perfs.extend([
                    p for p in symbol_perfs
                    if p["market_condition"] = = market_condition
                ])
                
            condition_performances[source] = source_condition_perfs
            
        # Check if we have enough data
        has_enough_data = all(
            len(perfs) >= self.min_samples
            for source, perfs in condition_performances.items()
            if source in self.source_weights
        )
        
        if not has_enough_data:
            return
            
        # Calculate performance-based adjustments
        adjustments = {}
        total_adjustment = 0
        
        for source in self.source_weights.keys():
            if source not in condition_performances or not condition_performances[source]:
                continue
                
            # Calculate average accuracy
            avg_accuracy = np.mean([p["accuracy"] for p in condition_performances[source]])
                
            # Calculate adjustment (positive if accuracy > 0.5, negative otherwise)
            current_weight = self.market_condition_weights[market_condition].get(source, 0.25)
            target_weight = current_weight * (1 + (avg_accuracy - 0.5) * self.learning_rate)
            
            # Ensure weight stays in reasonable range
            target_weight = max(0.05, min(0.5, target_weight))
            
            adjustments[source] = target_weight - current_weight
            total_adjustment += abs(adjustments[source])
            
        # Apply adjustments while maintaining sum of weights = 1
        if total_adjustment > 0:
            # Apply raw adjustments first
            old_weights = self.market_condition_weights[market_condition].copy()
            for source in self.market_condition_weights[market_condition]:
                if source in adjustments:
                    self.market_condition_weights[market_condition][source] += adjustments[source]
                    
            # Normalize weights to sum to 1
            weight_sum = sum(self.market_condition_weights[market_condition].values())
            for source in self.market_condition_weights[market_condition]:
                self.market_condition_weights[market_condition][source] /= weight_sum
                
            # Record weight adjustment
            if market_condition not in self.weight_history["by_market_condition"]:
                self.weight_history["by_market_condition"][market_condition] = []
                
            self.weight_history["by_market_condition"][market_condition].append({
                "timestamp": datetime.utcnow(),
                "old_weights": old_weights,
                "new_weights": self.market_condition_weights[market_condition].copy(),
                "adjustments": adjustments
            })
        
    def get_weights(self, 
                   symbol: Optional[str] = None,
                   market_condition: Optional[str] = None) -> Dict[str, float]:
        """Get the current source weights.
        
        Args:
            symbol: Optional symbol for symbol-specific weights
            market_condition: Optional market condition for condition-specific weights
            
        Returns:
            The current source weights
        """
        # Update weights if it's been more than a day
        if (datetime.utcnow() - self.last_update) > timedelta(days=1):
            self.update_weights()
            
        # Return appropriate weights
        if symbol and symbol in self.symbol_weights:
            return self.symbol_weights[symbol]
        elif market_condition and market_condition in self.market_condition_weights:
            return self.market_condition_weights[market_condition]
        else:
            return self.source_weights
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Performance metrics
        """
        return self.performance_metrics
        
    def visualize_weights(self, 
                         output_file: Optional[str] = None,
                         symbol: Optional[str] = None,
                         market_condition: Optional[str] = None) -> str:
        """Visualize weight changes over time.
        
        Args:
            output_file: Optional output file path
            symbol: Optional symbol for symbol-specific visualization
            market_condition: Optional market condition for condition-specific visualization
            
        Returns:
            Path to the generated visualization file
        """
        # Determine which weight history to use
        if symbol and symbol in self.weight_history["by_symbol"]:
            history = self.weight_history["by_symbol"][symbol]
            title = f"Weight Evolution for {symbol}"
            if not output_file:
                output_file = os.path.join(self.visualization_dir, f"weights_{symbol}.png")
        elif market_condition and market_condition in self.weight_history["by_market_condition"]:
            history = self.weight_history["by_market_condition"][market_condition]
            title = f"Weight Evolution for {market_condition} Market"
            if not output_file:
                output_file = os.path.join(self.visualization_dir, f"weights_{market_condition}.png")
        else:
            history = self.weight_history["global"]
            title = "Global Weight Evolution"
            if not output_file:
                output_file = os.path.join(self.visualization_dir, "weights_global.png")
                
        # Check if we have data to visualize
        if not history:
            return "No weight history available for visualization"
            
        # Extract data for plotting
        timestamps = [entry["timestamp"] for entry in history]
        sources = list(history[0]["new_weights"].keys())
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot weight evolution for each source
        for source in sources:
            weights = [entry["new_weights"][source] for entry in history]
            plt.plot(timestamps, weights, marker='o', label=source)
            
        # Add labels and legend
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
        
    def visualize_performance(self, 
                             output_file: Optional[str] = None,
                             by_source: bool = True,
                             by_symbol: bool = False,
                             by_market_condition: bool = False) -> str:
        """Visualize performance metrics.
        
        Args:
            output_file: Optional output file path
            by_source: Whether to show performance by source
            by_symbol: Whether to show performance by symbol
            by_market_condition: Whether to show performance by market condition
            
        Returns:
            Path to the generated visualization file
        """
        # Determine visualization type and output file
        viz_type = "performance"
        if by_source:
            viz_type += "_by_source"
        if by_symbol:
            viz_type += "_by_symbol"
        if by_market_condition:
            viz_type += "_by_market_condition"
            
        if not output_file:
            output_file = os.path.join(self.visualization_dir, f"{viz_type}.png")
            
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Plot performance metrics
        if by_source:
            # Performance by source
            sources = list(self.performance_metrics["by_source"].keys())
            accuracies = [self.performance_metrics["by_source"][s]["accuracy"] for s in sources]
            
            ax1 = fig.add_subplot(2, 1, 1)
            bars = ax1.bar(sources, accuracies)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
                
            ax1.set_title("Performance by Source")
            ax1.set_ylabel("Accuracy")
            ax1.set_ylim(0, 1)
            ax1.grid(axis='y')
            
        if by_market_condition and "by_market_condition" in self.performance_metrics:
            # Performance by market condition
            conditions = list(self.performance_metrics["by_market_condition"].keys())
            condition_accuracies = [self.performance_metrics["by_market_condition"][c]["accuracy"] for c in conditions]
            
            ax_pos = 2 if by_source else 1
            ax2 = fig.add_subplot(2, 1, ax_pos)
            bars = ax2.bar(conditions, condition_accuracies)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
                
            ax2.set_title("Performance by Market Condition")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0, 1)
            ax2.grid(axis='y')
            
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
        
    def export_weights(self, output_file: Optional[str] = None) -> str:
        """Export weights to a JSON file.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the exported file
        """
        if not output_file:
            output_file = os.path.join(self.visualization_dir, "weights_export.json")
            
        # Prepare data for export
        export_data = {
            "global_weights": self.source_weights,
            "symbol_weights": self.symbol_weights,
            "market_condition_weights": self.market_condition_weights,
            "performance_metrics": self.performance_metrics,
            "export_time": datetime.utcnow().isoformat()
        }
        
        # Export to JSON
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        return output_file
        
    def import_weights(self, input_file: str) -> bool:
        """Import weights from a JSON file.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            Whether the import was successful
        """
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
                
            # Import weights
            if "global_weights" in import_data:
                self.source_weights = import_data["global_weights"]
                
            if "symbol_weights" in import_data:
                self.symbol_weights = import_data["symbol_weights"]
                
            if "market_condition_weights" in import_data:
                self.market_condition_weights = import_data["market_condition_weights"]
                
            return True
        except Exception as e:
            print(f"Error importing weights: {e}")
            return False
