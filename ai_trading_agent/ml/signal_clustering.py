"""
Signal Clustering Module

This module provides clustering and similarity analysis for trading signals,
enabling pattern recognition and similar historical signal identification.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import json

from ..common.utils import get_logger

class SignalClusterAnalyzer:
    """
    Analyzes trading signals through clustering to identify patterns and similar signals.
    
    This helps in finding historical signals with similar characteristics and outcomes,
    which improves validation quality and enables pattern-based prediction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal cluster analyzer.
        
        Args:
            config: Configuration dictionary with parameters
                - cluster_method: Clustering method to use ('kmeans' or 'dbscan')
                - n_clusters: Number of clusters for KMeans
                - eps: Epsilon value for DBSCAN
                - min_samples: Minimum samples per cluster for DBSCAN
                - similarity_threshold: Threshold for considering signals similar
                - max_history_size: Maximum number of historical signals to store
        """
        self.logger = get_logger("SignalClusterAnalyzer")
        self.config = config or {}
        
        # Extract configuration
        self.cluster_method = self.config.get("cluster_method", "kmeans")
        self.n_clusters = self.config.get("n_clusters", 5)
        self.eps = self.config.get("eps", 0.5)
        self.min_samples = self.config.get("min_samples", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.clusterer = self._init_clusterer()
        self.cluster_labels = {}
        self.signal_history = []
        self.feature_weights = {}
        
        self.logger.info(f"SignalClusterAnalyzer initialized with {self.cluster_method} method")
    
    def _init_clusterer(self):
        """Initialize the clustering algorithm based on configuration."""
        if self.cluster_method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.cluster_method == "dbscan":
            return DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='euclidean'
            )
        else:
            self.logger.warning(f"Unknown cluster method: {self.cluster_method}, defaulting to KMeans")
            return KMeans(n_clusters=5, random_state=42)
    
    def add_signal_to_history(self, signal_features: Dict[str, float], outcome: Optional[bool] = None):
        """
        Add a signal and its features to the history for future reference.
        
        Args:
            signal_features: Dictionary of signal features
            outcome: Optional outcome of the signal (True=success, False=failure)
        """
        # Create a copy of features with timestamp
        entry = {
            "features": signal_features.copy(),
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "cluster": None  # Will be assigned during clustering
        }
        
        self.signal_history.append(entry)
        
        # Limit history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]
    
    def update_signal_outcome(self, signal_id: str, outcome: bool):
        """
        Update the outcome of a previously recorded signal.
        
        Args:
            signal_id: ID of the signal to update
            outcome: Whether the signal was successful (True) or not (False)
        """
        for entry in self.signal_history:
            if entry.get("signal_id") == signal_id:
                entry["outcome"] = outcome
                break
    
    def cluster_signals(self):
        """
        Perform clustering on historical signals to identify patterns.
        
        Returns:
            Dictionary mapping cluster labels to lists of signals
        """
        if not self.signal_history:
            self.logger.warning("No signals in history for clustering")
            return {}
            
        try:
            # Extract features from history
            feature_lists = []
            for entry in self.signal_history:
                features = entry["features"]
                if features:
                    feature_lists.append(list(features.values()))
            
            if not feature_lists:
                return {}
                
            # Convert to numpy array and scale
            X = np.array(feature_lists)
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform clustering
            cluster_labels = self.clusterer.fit_predict(X_scaled)
            
            # Store cluster assignments in history
            for i, entry in enumerate(self.signal_history):
                if i < len(cluster_labels):
                    entry["cluster"] = int(cluster_labels[i])
            
            # Group signals by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(self.signal_history[i])
            
            self.cluster_labels = clusters
            self.logger.info(f"Clustered {len(self.signal_history)} signals into {len(clusters)} clusters")
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in signal clustering: {str(e)}")
            return {}
    
    def find_similar_signals(self, features: Dict[str, float], history: List[Dict] = None) -> List[Dict]:
        """
        Find signals with similar features from history.
        
        Args:
            features: Dictionary of signal features to compare
            history: Optional custom history to search in (uses self.signal_history if None)
            
        Returns:
            List of similar signals with their similarity scores
        """
        if not features:
            return []
            
        search_history = history or self.signal_history
        if not search_history:
            return []
            
        try:
            # Extract feature values as a vector
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Create matrix of historical features
            history_vectors = []
            valid_entries = []
            
            for entry in search_history:
                entry_features = entry.get("features", {})
                if entry_features and len(entry_features) == len(features):
                    history_vectors.append(list(entry_features.values()))
                    valid_entries.append(entry)
            
            if not history_vectors:
                return []
                
            # Calculate similarities
            history_matrix = np.array(history_vectors)
            
            # Scale both the query and history vectors
            scaler = StandardScaler()
            history_scaled = scaler.fit_transform(history_matrix)
            query_scaled = scaler.transform(feature_vector)
            
            # Calculate distances and convert to similarity scores
            distances = euclidean_distances(query_scaled, history_scaled)[0]
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
            similarities = 1.0 - (distances / max_distance)
            
            # Apply feature weights if available
            if self.feature_weights:
                weighted_similarities = similarities.copy()
                for i, entry in enumerate(valid_entries):
                    feature_keys = list(entry.get("features", {}).keys())
                    weight_sum = sum(self.feature_weights.get(k, 1.0) for k in feature_keys)
                    weighted_score = sum(
                        similarities[i] * self.feature_weights.get(k, 1.0) 
                        for k in feature_keys
                    ) / weight_sum
                    weighted_similarities[i] = weighted_score
                similarities = weighted_similarities
            
            # Pair similarities with entries
            similar_signals = []
            for i, (entry, similarity) in enumerate(zip(valid_entries, similarities)):
                if similarity >= self.similarity_threshold:
                    signal_copy = entry.copy()
                    signal_copy["similarity"] = float(similarity)
                    similar_signals.append(signal_copy)
            
            # Sort by similarity (descending)
            similar_signals.sort(key=lambda x: x["similarity"], reverse=True)
            
            self.logger.info(f"Found {len(similar_signals)} similar signals above threshold {self.similarity_threshold}")
            return similar_signals
            
        except Exception as e:
            self.logger.error(f"Error finding similar signals: {str(e)}")
            return []
    
    def analyze_cluster_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of signals within each cluster.
        
        Returns:
            Dictionary with performance metrics for each cluster
        """
        if not self.cluster_labels:
            self.cluster_signals()
            
        if not self.cluster_labels:
            return {}
            
        performance = {}
        
        for label, signals in self.cluster_labels.items():
            # Filter signals with known outcomes
            signals_with_outcomes = [s for s in signals if s.get("outcome") is not None]
            
            if not signals_with_outcomes:
                continue
                
            # Calculate success rate
            success_count = sum(1 for s in signals_with_outcomes if s.get("outcome") is True)
            total_count = len(signals_with_outcomes)
            success_rate = success_count / total_count if total_count > 0 else 0
            
            # Store performance metrics
            performance[str(label)] = {
                "signal_count": total_count,
                "success_count": success_count,
                "success_rate": success_rate,
                "avg_similarity": np.mean([s.get("similarity", 0) for s in signals_with_outcomes])
            }
        
        return performance
    
    def update_feature_weights(self, weights: Dict[str, float]):
        """
        Update the weights used for feature importance in similarity calculations.
        
        Args:
            weights: Dictionary mapping feature names to their importance weights
        """
        self.feature_weights.update(weights)
        self.logger.info(f"Updated feature weights for {len(weights)} features")
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current clustering.
        
        Returns:
            Dictionary with cluster statistics
        """
        if not self.cluster_labels:
            return {}
            
        stats = {
            "cluster_count": len(self.cluster_labels),
            "total_signals": len(self.signal_history),
            "signals_per_cluster": {
                str(label): len(signals) for label, signals in self.cluster_labels.items()
            },
            "performance": self.analyze_cluster_performance()
        }
        
        return stats
