"""
Anomaly detection for log analysis.

This module provides functionality to detect anomalies in log data using statistical
and pattern-based approaches. It can identify unusual patterns, frequency changes,
and potentially problematic sequences in logs.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set

class AnomalyDetector:
    """Detect anomalies in log data using multiple detection methods."""
    
    def __init__(self, sensitivity: float = 0.95):
        """
        Initialize the anomaly detector.
        
        Args:
            sensitivity: Detection sensitivity (0.0-1.0), higher values detect more anomalies
        """
        self.sensitivity = sensitivity
        self.baseline = {}
        self.trained = False
    
    def train(self, logs: List[Dict[str, Any]]) -> None:
        """
        Train the detector on normal log data to establish baselines.
        
        Args:
            logs: List of log entries
        """
        if not logs:
            return
            
        # Extract components and levels
        components = [log.get('component', 'unknown') for log in logs if 'component' in log]
        levels = [log.get('level', 'info') for log in logs if 'level' in log]
        
        # Calculate baseline frequencies
        total_entries = len(logs)
        component_counts = Counter(components)
        level_counts = Counter(levels)
        
        # Calculate temporal patterns
        hourly_patterns = defaultdict(int)
        for log in logs:
            if 'timestamp' in log:
                try:
                    timestamp = log['timestamp']
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    hour = dt.hour
                    hourly_patterns[hour] += 1
                except (ValueError, TypeError):
                    continue
        
        # Calculate error message patterns
        error_messages = []
        for log in logs:
            if log.get('level') == 'error' and 'event' in log:
                error_messages.append(log['event'])
        error_pattern_counts = Counter(error_messages)
        
        # Store baseline data
        self.baseline = {
            'total_entries': total_entries,
            'component_frequencies': {comp: count/total_entries for comp, count in component_counts.items()},
            'level_frequencies': {level: count/total_entries for level, count in level_counts.items()},
            'hourly_patterns': {hour: count/total_entries for hour, count in hourly_patterns.items()},
            'error_patterns': error_pattern_counts,
            'avg_errors_per_hour': len(error_messages) / max(1, len(hourly_patterns))
        }
        
        self.trained = True
    
    def detect_anomalies(self, logs: List[Dict[str, Any]], window_size: int = 100) -> List[Dict[str, Any]]:
        """
        Detect anomalies in new log data.
        
        Args:
            logs: List of log entries to analyze
            window_size: Size of the window for rolling anomaly detection
            
        Returns:
            List of anomaly entries with explanation
        """
        if not self.trained or not logs:
            return []
        
        anomalies = []
        
        # Process in rolling windows
        for i in range(0, len(logs), window_size):
            window = logs[i:i+window_size]
            
            # Skip empty windows
            if not window:
                continue
            
            # Detect frequency anomalies
            level_anomalies = self._detect_level_anomalies(window)
            component_anomalies = self._detect_component_anomalies(window)
            temporal_anomalies = self._detect_temporal_anomalies(window)
            
            # Detect burst anomalies (sudden increase in errors)
            burst_anomalies = self._detect_error_bursts(window)
            
            # Combine all anomalies
            all_anomalies = level_anomalies + component_anomalies + temporal_anomalies + burst_anomalies
            anomalies.extend(all_anomalies)
        
        return anomalies
    
    def _detect_level_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in log level distribution."""
        anomalies = []
        
        # Count levels in current logs
        levels = [log.get('level', 'info') for log in logs if 'level' in log]
        level_counts = Counter(levels)
        total = len(logs)
        
        # Compare with baseline
        for level, count in level_counts.items():
            frequency = count / total
            baseline_freq = self.baseline['level_frequencies'].get(level, 0.01)
            
            # Calculate z-score for anomaly detection
            threshold = 3.0 * (1.0 + self.sensitivity)
            if level.lower() in ('error', 'critical'):
                # More sensitive to error level anomalies
                threshold *= 0.5
                
            # If frequency is significantly different from baseline
            if baseline_freq > 0 and frequency / baseline_freq > threshold:
                # Find all the anomalous logs with this level
                for log in logs:
                    if log.get('level') == level:
                        anomaly = log.copy()
                        anomaly["anomaly_type"] = 'level_frequency'
                        anomaly["anomaly_score"] = frequency / baseline_freq
                        anomaly["anomaly_explanation"] = f"Unusually high frequency of {level} logs"
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_component_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in component activity."""
        anomalies = []
        
        # Count components in current logs
        components = [log.get('component', 'unknown') for log in logs if 'component' in log]
        component_counts = Counter(components)
        total = len(logs)
        
        # Compare with baseline
        for component, count in component_counts.items():
            frequency = count / total
            baseline_freq = self.baseline['component_frequencies'].get(component, 0.01)
            
            # Calculate threshold for anomaly detection
            threshold = 3.0 * (1.0 + self.sensitivity)
            
            # If frequency is significantly different from baseline
            if baseline_freq > 0 and frequency / baseline_freq > threshold:
                # Find a representative log with this component that is most severe
                representative_log = None
                for log in logs:
                    if log.get('component') == component:
                        if representative_log is None or self._get_level_severity(log.get('level', 'info')) > self._get_level_severity(representative_log.get('level', 'info')):
                            representative_log = log
                
                if representative_log:
                    anomaly = representative_log.copy()
                    anomaly["anomaly_type"] = 'component_activity'
                    anomaly["anomaly_score"] = frequency / baseline_freq
                    anomaly["anomaly_explanation"] = f"Unusual activity from component {component}"
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_temporal_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in temporal patterns."""
        anomalies = []
        
        # Organize logs by hour
        hourly_logs = defaultdict(list)
        for log in logs:
            if 'timestamp' in log:
                try:
                    timestamp = log['timestamp']
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    hour = dt.hour
                    hourly_logs[hour].append(log)
                except (ValueError, TypeError):
                    continue
        
        # Check for unusual activity hours
        for hour, hour_logs in hourly_logs.items():
            observed_freq = len(hour_logs) / len(logs)
            baseline_freq = self.baseline['hourly_patterns'].get(hour, 0.04)  # 1/24 as default
            
            threshold = 4.0 * (1.0 + self.sensitivity)  # Higher threshold for temporal anomalies
            
            if baseline_freq > 0 and observed_freq / baseline_freq > threshold:
                # Find the most severe log in this hour
                most_severe_log = max(hour_logs, key=lambda x: self._get_level_severity(x.get('level', 'info')))
                anomaly = most_severe_log.copy()
                anomaly["anomaly_type"] = 'temporal_pattern'
                anomaly["anomaly_score"] = observed_freq / baseline_freq
                anomaly["anomaly_explanation"] = f"Unusual activity during hour {hour}"
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_error_bursts(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect bursts of errors."""
        anomalies = []
        
        # Count errors
        error_logs = [log for log in logs if log.get('level') in ('error', 'critical')]
        
        if not error_logs:
            return []
            
        # Get the time span of the window
        timestamps = []
        for log in logs:
            if 'timestamp' in log:
                try:
                    timestamp = log['timestamp']
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    timestamps.append(dt)
                except (ValueError, TypeError):
                    continue
        
        if not timestamps:
            return []
            
        # Calculate the time span in hours
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
        time_span = max(time_span, 0.01)  # Prevent division by zero
        
        # Calculate error rate
        error_rate = len(error_logs) / time_span
        baseline_rate = self.baseline.get('avg_errors_per_hour', 0.1)
        
        # Check if error rate is significantly higher
        threshold = 2.0 * (1.0 + self.sensitivity)
        if baseline_rate > 0 and error_rate / baseline_rate > threshold:
            # Group errors by type/message
            error_types = Counter()
            for log in error_logs:
                message = log.get('event', '')
                error_types[message] += 1
            
            # Find the most common error
            most_common_error = error_types.most_common(1)[0][0] if error_types else "Unknown error"
            
            # Find a representative error log
            representative_log = next((log for log in error_logs if log.get('event') == most_common_error), error_logs[0])
            
            anomaly = representative_log.copy()
            anomaly["anomaly_type"] = 'error_burst'
            anomaly["anomaly_score"] = error_rate / baseline_rate
            anomaly["anomaly_explanation"] = f"Burst of errors detected ({len(error_logs)} errors in {time_span:.2f} hours)"
            anomaly["related_count"] = len(error_logs)
            anomalies.append(anomaly)
        
        return anomalies
    
    def _get_level_severity(self, level: str) -> int:
        """Get numeric severity for a log level."""
        level_map = {
            'debug': 0,
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }
        return level_map.get(level.lower(), 1)
