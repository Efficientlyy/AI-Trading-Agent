"""
Validation Engine

This module provides a class for validating data against rules,
including range validation, temporal validation, cross-field validation,
and anomaly detection.
"""

import logging
import json
import time
import datetime
import statistics
import math
import re
from typing import Dict, List, Any, Optional, Union, Callable, Set
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("validation_engine")

class ValidationEngine:
    """
    Validates data against rules and detects anomalies.
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialize the validation engine.
        
        Args:
            rules_file: Path to a JSON file containing validation rules (optional)
        """
        self.rules = {}
        self.validation_history = {}
        self.anomaly_history = {}
        self.lock = Lock()
        self.stats = {
            "validations_performed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "anomalies_detected": 0,
            "start_time": time.time()
        }
        
        # Load rules from file if provided
        if rules_file:
            self.load_rules(rules_file)
    
    def load_rules(self, rules_file: str) -> bool:
        """
        Load validation rules from a JSON file.
        
        Args:
            rules_file: Path to the JSON file
            
        Returns:
            True if rules were loaded successfully, False otherwise
        """
        try:
            with open(rules_file, 'r') as f:
                self.rules = json.load(f)
            logger.info(f"Loaded validation rules from {rules_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading validation rules: {e}")
            return False
    
    def save_rules(self, rules_file: str) -> bool:
        """
        Save validation rules to a JSON file.
        
        Args:
            rules_file: Path to the JSON file
            
        Returns:
            True if rules were saved successfully, False otherwise
        """
        try:
            with open(rules_file, 'w') as f:
                json.dump(self.rules, f, indent=2)
            logger.info(f"Saved validation rules to {rules_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving validation rules: {e}")
            return False
    
    def add_rule(self, rule_id: str, rule_type: str, rule_config: Dict[str, Any]) -> bool:
        """
        Add a validation rule.
        
        Args:
            rule_id: The rule ID
            rule_type: The rule type (range, temporal, cross-field, anomaly)
            rule_config: The rule configuration
            
        Returns:
            True if the rule was added successfully, False otherwise
        """
        with self.lock:
            self.rules[rule_id] = {
                "type": rule_type,
                "config": rule_config,
                "enabled": True,
                "created_at": time.time()
            }
        
        logger.info(f"Added validation rule: {rule_id} ({rule_type})")
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a validation rule.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            True if the rule was removed successfully, False otherwise
        """
        with self.lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed validation rule: {rule_id}")
                return True
            else:
                logger.warning(f"Rule not found: {rule_id}")
                return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a validation rule.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            True if the rule was enabled successfully, False otherwise
        """
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id]["enabled"] = True
                logger.info(f"Enabled validation rule: {rule_id}")
                return True
            else:
                logger.warning(f"Rule not found: {rule_id}")
                return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable a validation rule.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            True if the rule was disabled successfully, False otherwise
        """
        with self.lock:
            if rule_id in self.rules:
                self.rules[rule_id]["enabled"] = False
                logger.info(f"Disabled validation rule: {rule_id}")
                return True
            else:
                logger.warning(f"Rule not found: {rule_id}")
                return False
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate data against all enabled rules.
        
        Args:
            data: The data to validate
            context: Additional context for validation (optional)
            
        Returns:
            Dict containing validation results
        """
        results = {
            "valid": True,
            "rule_results": {},
            "anomalies": [],
            "timestamp": time.time()
        }
        
        with self.lock:
            # Increment validation count
            self.stats["validations_performed"] += 1
            
            # Get enabled rules
            enabled_rules = {rule_id: rule for rule_id, rule in self.rules.items() if rule.get("enabled", True)}
        
        # Validate against each rule
        for rule_id, rule in enabled_rules.items():
            rule_type = rule["type"]
            rule_config = rule["config"]
            
            # Validate based on rule type
            if rule_type == "range":
                result = self._validate_range(data, rule_config)
            elif rule_type == "temporal":
                result = self._validate_temporal(data, rule_config, context)
            elif rule_type == "cross-field":
                result = self._validate_cross_field(data, rule_config)
            elif rule_type == "anomaly":
                result = self._detect_anomaly(data, rule_config, context)
            else:
                logger.warning(f"Unknown rule type: {rule_type}")
                result = {"valid": True, "reason": f"Unknown rule type: {rule_type}"}
            
            # Store rule result
            results["rule_results"][rule_id] = result
            
            # Update overall validity
            if not result["valid"]:
                results["valid"] = False
            
            # Add anomaly if detected
            if rule_type == "anomaly" and not result["valid"]:
                results["anomalies"].append({
                    "rule_id": rule_id,
                    "field": rule_config.get("field"),
                    "reason": result["reason"],
                    "value": result.get("value"),
                    "expected": result.get("expected")
                })
        
        # Update validation history
        self._update_validation_history(data, results)
        
        # Update stats
        with self.lock:
            if results["valid"]:
                self.stats["validations_passed"] += 1
            else:
                self.stats["validations_failed"] += 1
            
            if results["anomalies"]:
                self.stats["anomalies_detected"] += len(results["anomalies"])
        
        return results
    
    def _validate_range(self, data: Dict[str, Any], rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a value against a range rule.
        
        Args:
            data: The data to validate
            rule_config: The rule configuration
            
        Returns:
            Dict containing validation result
        """
        field = rule_config.get("field")
        if not field or field not in data:
            return {"valid": True, "reason": f"Field not found: {field}"}
        
        value = data[field]
        
        # Skip validation if value is not numeric
        if not isinstance(value, (int, float)):
            return {"valid": True, "reason": f"Value is not numeric: {value}"}
        
        # Check minimum value
        min_value = rule_config.get("min")
        if min_value is not None and value < min_value:
            return {
                "valid": False,
                "reason": f"Value below minimum: {value} < {min_value}",
                "value": value,
                "expected": f">= {min_value}"
            }
        
        # Check maximum value
        max_value = rule_config.get("max")
        if max_value is not None and value > max_value:
            return {
                "valid": False,
                "reason": f"Value above maximum: {value} > {max_value}",
                "value": value,
                "expected": f"<= {max_value}"
            }
        
        # Check percentage change if previous value is available
        if "percentage_change" in rule_config and "previous_value" in rule_config:
            previous_value = rule_config["previous_value"]
            max_percentage_change = rule_config["percentage_change"]
            
            if previous_value != 0:
                percentage_change = abs((value - previous_value) / previous_value) * 100
                
                if percentage_change > max_percentage_change:
                    return {
                        "valid": False,
                        "reason": f"Percentage change too high: {percentage_change:.2f}% > {max_percentage_change}%",
                        "value": value,
                        "previous_value": previous_value,
                        "percentage_change": percentage_change,
                        "expected": f"<= {max_percentage_change}%"
                    }
        
        return {"valid": True}
    
    def _validate_temporal(self, data: Dict[str, Any], rule_config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate temporal data.
        
        Args:
            data: The data to validate
            rule_config: The rule configuration
            context: Additional context for validation
            
        Returns:
            Dict containing validation result
        """
        field = rule_config.get("field")
        if not field or field not in data:
            return {"valid": True, "reason": f"Field not found: {field}"}
        
        value = data[field]
        
        # Convert string to datetime if needed
        if isinstance(value, str):
            try:
                value = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return {
                    "valid": False,
                    "reason": f"Invalid datetime format: {value}",
                    "value": value
                }
        
        # Skip validation if value is not a datetime
        if not isinstance(value, (datetime.datetime, datetime.date)):
            return {"valid": True, "reason": f"Value is not a datetime: {value}"}
        
        # Check if timestamp is in the future
        if rule_config.get("no_future", False):
            now = datetime.datetime.now(datetime.timezone.utc)
            if value > now:
                return {
                    "valid": False,
                    "reason": f"Timestamp is in the future: {value} > {now}",
                    "value": value,
                    "expected": f"<= {now}"
                }
        
        # Check if timestamp is too old
        max_age = rule_config.get("max_age")
        if max_age is not None:
            now = datetime.datetime.now(datetime.timezone.utc)
            if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
                value = datetime.datetime.combine(value, datetime.time.min).replace(tzinfo=datetime.timezone.utc)
            
            age = now - value
            max_age_seconds = max_age * 3600  # Convert hours to seconds
            
            if age.total_seconds() > max_age_seconds:
                return {
                    "valid": False,
                    "reason": f"Timestamp too old: {age.total_seconds() / 3600:.2f} hours > {max_age} hours",
                    "value": value,
                    "expected": f">= {now - datetime.timedelta(seconds=max_age_seconds)}"
                }
        
        # Check sequence if previous timestamp is available
        if "previous_timestamp" in rule_config:
            previous_timestamp = rule_config["previous_timestamp"]
            
            if isinstance(previous_timestamp, str):
                try:
                    previous_timestamp = datetime.datetime.fromisoformat(previous_timestamp.replace('Z', '+00:00'))
                except ValueError:
                    return {"valid": True, "reason": f"Invalid previous timestamp format: {previous_timestamp}"}
            
            if value < previous_timestamp:
                return {
                    "valid": False,
                    "reason": f"Timestamp out of sequence: {value} < {previous_timestamp}",
                    "value": value,
                    "expected": f">= {previous_timestamp}"
                }
        
        return {"valid": True}
    
    def _validate_cross_field(self, data: Dict[str, Any], rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate relationships between fields.
        
        Args:
            data: The data to validate
            rule_config: The rule configuration
            
        Returns:
            Dict containing validation result
        """
        field1 = rule_config.get("field1")
        field2 = rule_config.get("field2")
        
        if not field1 or not field2:
            return {"valid": True, "reason": "Missing field names in rule configuration"}
        
        if field1 not in data or field2 not in data:
            return {"valid": True, "reason": f"Field not found: {field1 if field1 not in data else field2}"}
        
        value1 = data[field1]
        value2 = data[field2]
        
        # Skip validation if values are not comparable
        if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
            return {"valid": True, "reason": f"Values are not comparable: {value1} ({type(value1).__name__}), {value2} ({type(value2).__name__})"}
        
        # Check relationship
        relationship = rule_config.get("relationship", "==")
        
        if relationship == "==":
            if value1 != value2:
                return {
                    "valid": False,
                    "reason": f"{field1} != {field2}: {value1} != {value2}",
                    "value1": value1,
                    "value2": value2,
                    "expected": f"{field1} == {field2}"
                }
        elif relationship == "!=":
            if value1 == value2:
                return {
                    "valid": False,
                    "reason": f"{field1} == {field2}: {value1} == {value2}",
                    "value1": value1,
                    "value2": value2,
                    "expected": f"{field1} != {field2}"
                }
        elif relationship == "<":
            if value1 >= value2:
                return {
                    "valid": False,
                    "reason": f"{field1} >= {field2}: {value1} >= {value2}",
                    "value1": value1,
                    "value2": value2,
                    "expected": f"{field1} < {field2}"
                }
        elif relationship == "<=":
            if value1 > value2:
                return {
                    "valid": False,
                    "reason": f"{field1} > {field2}: {value1} > {value2}",
                    "value1": value1,
                    "value2": value2,
                    "expected": f"{field1} <= {field2}"
                }
        elif relationship == ">":
            if value1 <= value2:
                return {
                    "valid": False,
                    "reason": f"{field1} <= {field2}: {value1} <= {value2}",
                    "value1": value1,
                    "value2": value2,
                    "expected": f"{field1} > {field2}"
                }
        elif relationship == ">=":
            if value1 < value2:
                return {
                    "valid": False,
                    "reason": f"{field1} < {field2}: {value1} < {value2}",
                    "value1": value1,
                    "value2": value2,
                    "expected": f"{field1} >= {field2}"
                }
        elif relationship == "ratio":
            # Check if the ratio between the fields is within the specified range
            ratio_min = rule_config.get("ratio_min")
            ratio_max = rule_config.get("ratio_max")
            
            if value2 == 0:
                return {
                    "valid": False,
                    "reason": f"Cannot calculate ratio: {field2} is zero",
                    "value1": value1,
                    "value2": value2
                }
            
            ratio = value1 / value2
            
            if ratio_min is not None and ratio < ratio_min:
                return {
                    "valid": False,
                    "reason": f"Ratio too low: {ratio:.2f} < {ratio_min}",
                    "value1": value1,
                    "value2": value2,
                    "ratio": ratio,
                    "expected": f">= {ratio_min}"
                }
            
            if ratio_max is not None and ratio > ratio_max:
                return {
                    "valid": False,
                    "reason": f"Ratio too high: {ratio:.2f} > {ratio_max}",
                    "value1": value1,
                    "value2": value2,
                    "ratio": ratio,
                    "expected": f"<= {ratio_max}"
                }
        else:
            return {
                "valid": False,
                "reason": f"Unknown relationship: {relationship}",
                "value1": value1,
                "value2": value2
            }
        
        return {"valid": True}
    
    def _detect_anomaly(self, data: Dict[str, Any], rule_config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect anomalies in data.
        
        Args:
            data: The data to validate
            rule_config: The rule configuration
            context: Additional context for validation
            
        Returns:
            Dict containing validation result
        """
        field = rule_config.get("field")
        if not field or field not in data:
            return {"valid": True, "reason": f"Field not found: {field}"}
        
        value = data[field]
        
        # Skip validation if value is not numeric
        if not isinstance(value, (int, float)):
            return {"valid": True, "reason": f"Value is not numeric: {value}"}
        
        # Get historical values
        historical_values = self._get_historical_values(field)
        
        # Skip if not enough historical data
        min_history = rule_config.get("min_history", 10)
        if len(historical_values) < min_history:
            return {"valid": True, "reason": f"Not enough historical data: {len(historical_values)} < {min_history}"}
        
        # Detect anomaly based on method
        method = rule_config.get("method", "z-score")
        
        if method == "z-score":
            # Z-score method
            threshold = rule_config.get("threshold", 3.0)
            
            # Calculate mean and standard deviation
            mean = statistics.mean(historical_values)
            stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            
            # Skip if standard deviation is zero
            if stdev == 0:
                return {"valid": True, "reason": "Standard deviation is zero"}
            
            # Calculate z-score
            z_score = abs((value - mean) / stdev)
            
            if z_score > threshold:
                return {
                    "valid": False,
                    "reason": f"Z-score too high: {z_score:.2f} > {threshold}",
                    "value": value,
                    "mean": mean,
                    "stdev": stdev,
                    "z_score": z_score,
                    "threshold": threshold
                }
        elif method == "iqr":
            # Interquartile Range method
            threshold = rule_config.get("threshold", 1.5)
            
            # Calculate quartiles
            sorted_values = sorted(historical_values)
            q1_index = int(len(sorted_values) * 0.25)
            q3_index = int(len(sorted_values) * 0.75)
            q1 = sorted_values[q1_index]
            q3 = sorted_values[q3_index]
            iqr = q3 - q1
            
            # Calculate bounds
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            if value < lower_bound:
                return {
                    "valid": False,
                    "reason": f"Value below lower bound: {value} < {lower_bound}",
                    "value": value,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }
            
            if value > upper_bound:
                return {
                    "valid": False,
                    "reason": f"Value above upper bound: {value} > {upper_bound}",
                    "value": value,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }
        elif method == "mad":
            # Median Absolute Deviation method
            threshold = rule_config.get("threshold", 3.0)
            
            # Calculate median
            median = statistics.median(historical_values)
            
            # Calculate MAD
            deviations = [abs(x - median) for x in historical_values]
            mad = statistics.median(deviations)
            
            # Skip if MAD is zero
            if mad == 0:
                return {"valid": True, "reason": "Median Absolute Deviation is zero"}
            
            # Calculate score
            score = abs(value - median) / mad
            
            if score > threshold:
                return {
                    "valid": False,
                    "reason": f"MAD score too high: {score:.2f} > {threshold}",
                    "value": value,
                    "median": median,
                    "mad": mad,
                    "score": score,
                    "threshold": threshold
                }
        else:
            return {
                "valid": False,
                "reason": f"Unknown anomaly detection method: {method}",
                "value": value
            }
        
        return {"valid": True}
    
    def _get_historical_values(self, field: str) -> List[float]:
        """
        Get historical values for a field.
        
        Args:
            field: The field name
            
        Returns:
            List of historical values
        """
        with self.lock:
            if field not in self.validation_history:
                return []
            
            return [entry["value"] for entry in self.validation_history[field]]
    
    def _update_validation_history(self, data: Dict[str, Any], results: Dict[str, Any]):
        """
        Update validation history with new data.
        
        Args:
            data: The validated data
            results: The validation results
        """
        timestamp = results["timestamp"]
        
        with self.lock:
            # Update validation history for each field
            for field, value in data.items():
                if isinstance(value, (int, float)):
                    if field not in self.validation_history:
                        self.validation_history[field] = []
                    
                    # Add to history
                    self.validation_history[field].append({
                        "timestamp": timestamp,
                        "value": value
                    })
                    
                    # Limit history size
                    max_history = 1000
                    if len(self.validation_history[field]) > max_history:
                        self.validation_history[field] = self.validation_history[field][-max_history:]
            
            # Update anomaly history
            for anomaly in results["anomalies"]:
                field = anomaly["field"]
                
                if field not in self.anomaly_history:
                    self.anomaly_history[field] = []
                
                # Add to history
                self.anomaly_history[field].append({
                    "timestamp": timestamp,
                    "rule_id": anomaly["rule_id"],
                    "reason": anomaly["reason"],
                    "value": anomaly.get("value")
                })
                
                # Limit history size
                max_history = 100
                if len(self.anomaly_history[field]) > max_history:
                    self.anomaly_history[field] = self.anomaly_history[field][-max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dict containing validation statistics
        """
        with self.lock:
            stats = self.stats.copy()
            stats["uptime"] = time.time() - stats["start_time"]
            stats["rules_count"] = len(self.rules)
            stats["enabled_rules_count"] = len([rule for rule in self.rules.values() if rule.get("enabled", True)])
            stats["fields_with_history"] = len(self.validation_history)
            stats["fields_with_anomalies"] = len(self.anomaly_history)
            return stats
    
    def get_anomalies(self, field: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get detected anomalies.
        
        Args:
            field: The field to get anomalies for (optional)
            limit: Maximum number of anomalies to return
            
        Returns:
            List of anomalies
        """
        with self.lock:
            if field:
                # Get anomalies for a specific field
                if field not in self.anomaly_history:
                    return []
                
                return self.anomaly_history[field][-limit:]
            else:
                # Get anomalies for all fields
                anomalies = []
                
                for field, field_anomalies in self.anomaly_history.items():
                    anomalies.extend([{**anomaly, "field": field} for anomaly in field_anomalies])
                
                # Sort by timestamp (newest first)
                anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
                
                return anomalies[:limit]
    
    def get_rules(self) -> Dict[str, Any]:
        """
        Get all validation rules.
        
        Returns:
            Dict containing validation rules
        """
        with self.lock:
            return self.rules.copy()