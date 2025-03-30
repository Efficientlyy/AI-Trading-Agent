"""
Transformation Pipeline

This module provides a class for transforming data through a pipeline of operations,
including normalization, format standardization, enrichment, and optimization.
"""

import logging
import json
import time
import datetime
import re
import copy
from typing import Dict, List, Any, Optional, Union, Callable
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("transformation_pipeline")

class TransformationPipeline:
    """
    Transforms data through a pipeline of operations.
    """
    
    def __init__(self, pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transformation pipeline.
        
        Args:
            pipeline_config: Pipeline configuration (optional)
        """
        self.transformers = {}
        self.pipelines = {}
        self.lock = Lock()
        self.stats = {
            "transformations_performed": 0,
            "transformations_succeeded": 0,
            "transformations_failed": 0,
            "start_time": time.time()
        }
        
        # Register built-in transformers
        self._register_built_in_transformers()
        
        # Load pipeline configuration if provided
        if pipeline_config:
            self.configure(pipeline_config)
    
    def _register_built_in_transformers(self):
        """
        Register built-in transformers.
        """
        # Field transformers
        self.register_transformer("rename_field", self._transform_rename_field)
        self.register_transformer("remove_field", self._transform_remove_field)
        self.register_transformer("copy_field", self._transform_copy_field)
        
        # Value transformers
        self.register_transformer("convert_type", self._transform_convert_type)
        self.register_transformer("format_number", self._transform_format_number)
        self.register_transformer("format_date", self._transform_format_date)
        
        # Structure transformers
        self.register_transformer("flatten", self._transform_flatten)
        self.register_transformer("nest", self._transform_nest)
        
        # Array transformers
        self.register_transformer("filter_array", self._transform_filter_array)
        self.register_transformer("sort_array", self._transform_sort_array)
        
        # Enrichment transformers
        self.register_transformer("calculate_field", self._transform_calculate_field)
        self.register_transformer("default_value", self._transform_default_value)
        
        # Optimization transformers
        self.register_transformer("round_number", self._transform_round_number)
        self.register_transformer("truncate_string", self._transform_truncate_string)
    
    def register_transformer(self, name: str, transformer_func: Callable) -> bool:
        """
        Register a transformer function.
        
        Args:
            name: The transformer name
            transformer_func: The transformer function
            
        Returns:
            True if the transformer was registered successfully, False otherwise
        """
        with self.lock:
            if name in self.transformers:
                logger.warning(f"Transformer already exists: {name}")
                return False
            
            self.transformers[name] = transformer_func
            logger.info(f"Registered transformer: {name}")
            return True
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the transformation pipeline.
        
        Args:
            config: The pipeline configuration
            
        Returns:
            True if the pipeline was configured successfully, False otherwise
        """
        try:
            # Configure pipelines
            pipelines = config.get("pipelines", {})
            
            with self.lock:
                self.pipelines = {}
                
                for pipeline_id, pipeline_config in pipelines.items():
                    self.pipelines[pipeline_id] = pipeline_config
            
            logger.info(f"Configured {len(self.pipelines)} pipelines")
            return True
        except Exception as e:
            logger.error(f"Error configuring pipeline: {e}")
            return False
    
    def transform(self, data: Any, pipeline_id: Optional[str] = None, transformers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Transform data through a pipeline.
        
        Args:
            data: The data to transform
            pipeline_id: The pipeline ID to use (optional)
            transformers: List of transformers to apply (optional)
            
        Returns:
            Dict containing the transformation result
        """
        # Create a deep copy of the data to avoid modifying the original
        result = copy.deepcopy(data)
        
        # Get transformers to apply
        transformers_to_apply = []
        
        if pipeline_id:
            # Get pipeline configuration
            with self.lock:
                if pipeline_id not in self.pipelines:
                    return {
                        "success": False,
                        "error": f"Pipeline not found: {pipeline_id}",
                        "data": data
                    }
                
                pipeline_config = self.pipelines[pipeline_id]
                transformers_to_apply = pipeline_config.get("transformers", [])
        elif transformers:
            # Use provided transformers
            transformers_to_apply = transformers
        else:
            return {
                "success": False,
                "error": "No pipeline or transformers specified",
                "data": data
            }
        
        # Apply transformers
        transformation_results = []
        success = True
        error = None
        
        for transformer_config in transformers_to_apply:
            transformer_name = transformer_config.get("name")
            transformer_params = transformer_config.get("params", {})
            
            if not transformer_name:
                transformation_results.append({
                    "name": "unknown",
                    "success": False,
                    "error": "Transformer name not specified"
                })
                success = False
                error = "Transformer name not specified"
                continue
            
            with self.lock:
                if transformer_name not in self.transformers:
                    transformation_results.append({
                        "name": transformer_name,
                        "success": False,
                        "error": f"Transformer not found: {transformer_name}"
                    })
                    success = False
                    error = f"Transformer not found: {transformer_name}"
                    continue
                
                transformer_func = self.transformers[transformer_name]
            
            try:
                # Apply transformer
                transformer_result = transformer_func(result, **transformer_params)
                
                if isinstance(transformer_result, dict) and "success" in transformer_result:
                    # Transformer returned a result object
                    if transformer_result["success"]:
                        result = transformer_result["data"]
                    else:
                        success = False
                        error = transformer_result.get("error", f"Transformer failed: {transformer_name}")
                    
                    transformation_results.append({
                        "name": transformer_name,
                        "success": transformer_result["success"],
                        "error": transformer_result.get("error")
                    })
                else:
                    # Transformer returned the transformed data directly
                    result = transformer_result
                    
                    transformation_results.append({
                        "name": transformer_name,
                        "success": True
                    })
            except Exception as e:
                logger.error(f"Error applying transformer {transformer_name}: {e}")
                
                transformation_results.append({
                    "name": transformer_name,
                    "success": False,
                    "error": str(e)
                })
                
                success = False
                error = str(e)
        
        # Update stats
        with self.lock:
            self.stats["transformations_performed"] += 1
            
            if success:
                self.stats["transformations_succeeded"] += 1
            else:
                self.stats["transformations_failed"] += 1
        
        return {
            "success": success,
            "error": error,
            "data": result,
            "transformers": transformation_results
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get transformation statistics.
        
        Returns:
            Dict containing transformation statistics
        """
        with self.lock:
            stats = self.stats.copy()
            stats["uptime"] = time.time() - stats["start_time"]
            stats["transformers_count"] = len(self.transformers)
            stats["pipelines_count"] = len(self.pipelines)
            return stats
    
    def get_transformers(self) -> Dict[str, Any]:
        """
        Get registered transformers.
        
        Returns:
            Dict containing registered transformers
        """
        with self.lock:
            return {name: {"name": name} for name in self.transformers.keys()}
    
    def get_pipelines(self) -> Dict[str, Any]:
        """
        Get configured pipelines.
        
        Returns:
            Dict containing configured pipelines
        """
        with self.lock:
            return self.pipelines.copy()
    
    # Transformer implementations
    
    def _transform_rename_field(self, data: Dict[str, Any], source: str, target: str) -> Dict[str, Any]:
        """Rename a field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if source in data:
            data[target] = data[source]
            del data[source]
        
        return data
    
    def _transform_remove_field(self, data: Dict[str, Any], field: str) -> Dict[str, Any]:
        """Remove a field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field in data:
            del data[field]
        
        return data
    
    def _transform_copy_field(self, data: Dict[str, Any], source: str, target: str) -> Dict[str, Any]:
        """Copy a field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if source in data:
            data[target] = copy.deepcopy(data[source])
        
        return data
    
    def _transform_convert_type(self, data: Dict[str, Any], field: str, type_name: str) -> Dict[str, Any]:
        """Convert a field to a different type."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        try:
            if type_name == "string":
                data[field] = str(value)
            elif type_name == "number":
                if isinstance(value, str):
                    value = re.sub(r"[^\d.-]", "", value)
                data[field] = float(value)
            elif type_name == "integer":
                if isinstance(value, str):
                    value = re.sub(r"[^\d-]", "", value)
                data[field] = int(float(value))
            elif type_name == "boolean":
                if isinstance(value, str):
                    value = value.lower()
                    data[field] = value in ("true", "yes", "1", "y", "t")
                else:
                    data[field] = bool(value)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error converting {field} to {type_name}: {e}",
                "data": data
            }
        
        return data
    
    def _transform_format_number(self, data: Dict[str, Any], field: str, precision: int = 2) -> Dict[str, Any]:
        """Format a number field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        try:
            if not isinstance(value, (int, float)):
                value = float(value)
            
            # Round to specified precision
            value = round(value, precision)
            
            # Format with commas for thousands and period for decimal
            data[field] = f"{value:,.{precision}f}"
        except Exception as e:
            return {
                "success": False,
                "error": f"Error formatting {field}: {e}",
                "data": data
            }
        
        return data
    
    def _transform_format_date(self, data: Dict[str, Any], field: str, format: str = "%Y-%m-%d %H:%M:%S") -> Dict[str, Any]:
        """Format a date field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        try:
            if isinstance(value, (int, float)):
                # Assume timestamp
                dt = datetime.datetime.fromtimestamp(value)
            elif isinstance(value, str):
                # Try to parse as ISO format
                dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            elif isinstance(value, datetime.datetime):
                dt = value
            else:
                return {
                    "success": False,
                    "error": f"Cannot format {field} as date: unsupported type",
                    "data": data
                }
            
            # Format date
            data[field] = dt.strftime(format)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error formatting {field} as date: {e}",
                "data": data
            }
        
        return data
    
    def _transform_flatten(self, data: Dict[str, Any], prefix: str = "", separator: str = "_") -> Dict[str, Any]:
        """Flatten a nested structure."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        result = {}
        
        def _flatten(obj, current_prefix):
            for key, value in obj.items():
                new_key = f"{current_prefix}{separator}{key}" if current_prefix else key
                
                if isinstance(value, dict):
                    _flatten(value, new_key)
                else:
                    result[new_key] = value
        
        _flatten(data, prefix)
        
        return result
    
    def _transform_nest(self, data: Dict[str, Any], fields: List[str], target: str) -> Dict[str, Any]:
        """Nest fields under a target field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        # Create nested object
        nested = {}
        
        for field in fields:
            if field in data:
                nested[field] = data[field]
                del data[field]
        
        # Add nested object to data
        data[target] = nested
        
        return data
    
    def _transform_filter_array(self, data: Dict[str, Any], field: str, condition: Dict[str, Any]) -> Dict[str, Any]:
        """Filter an array field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        if not isinstance(value, list):
            return {
                "success": False,
                "error": f"Field {field} is not an array",
                "data": data
            }
        
        # Get condition parameters
        field_name = condition.get("field")
        operator = condition.get("operator", "eq")
        compare_value = condition.get("value")
        
        if not field_name:
            return {
                "success": False,
                "error": "Filter condition must specify a field",
                "data": data
            }
        
        # Filter array
        filtered = []
        
        for item in value:
            if not isinstance(item, dict) or field_name not in item:
                continue
            
            item_value = item[field_name]
            
            if operator == "eq" and item_value == compare_value:
                filtered.append(item)
            elif operator == "ne" and item_value != compare_value:
                filtered.append(item)
            elif operator == "gt" and item_value > compare_value:
                filtered.append(item)
            elif operator == "gte" and item_value >= compare_value:
                filtered.append(item)
            elif operator == "lt" and item_value < compare_value:
                filtered.append(item)
            elif operator == "lte" and item_value <= compare_value:
                filtered.append(item)
        
        data[field] = filtered
        
        return data
    
    def _transform_sort_array(self, data: Dict[str, Any], field: str, sort_field: str, order: str = "asc") -> Dict[str, Any]:
        """Sort an array field."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        if not isinstance(value, list):
            return {
                "success": False,
                "error": f"Field {field} is not an array",
                "data": data
            }
        
        # Sort array
        try:
            reverse = order.lower() == "desc"
            
            data[field] = sorted(
                value,
                key=lambda x: x[sort_field] if isinstance(x, dict) and sort_field in x else None,
                reverse=reverse
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Error sorting array: {e}",
                "data": data
            }
        
        return data
    
    def _transform_calculate_field(self, data: Dict[str, Any], target: str, expression: str) -> Dict[str, Any]:
        """Calculate a field value using an expression."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        # Simple expression evaluation (for demonstration)
        try:
            # Create a safe evaluation context with only the data fields
            context = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, context)
            
            # Set target field
            data[target] = result
        except Exception as e:
            return {
                "success": False,
                "error": f"Error calculating field: {e}",
                "data": data
            }
        
        return data
    
    def _transform_default_value(self, data: Dict[str, Any], field: str, default_value: Any) -> Dict[str, Any]:
        """Set a default value for a field if it doesn't exist or is None."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data or data[field] is None:
            data[field] = default_value
        
        return data
    
    def _transform_round_number(self, data: Dict[str, Any], field: str, precision: int = 2) -> Dict[str, Any]:
        """Round a number field to a specified precision."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        try:
            if not isinstance(value, (int, float)):
                value = float(value)
            
            data[field] = round(value, precision)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error rounding {field}: {e}",
                "data": data
            }
        
        return data
    
    def _transform_truncate_string(self, data: Dict[str, Any], field: str, max_length: int, suffix: str = "...") -> Dict[str, Any]:
        """Truncate a string field to a specified maximum length."""
        if not isinstance(data, dict):
            return {"success": False, "error": "Data is not a dictionary", "data": data}
        
        if field not in data:
            return data
        
        value = data[field]
        
        try:
            if not isinstance(value, str):
                value = str(value)
            
            if len(value) > max_length:
                data[field] = value[:max_length - len(suffix)] + suffix
        except Exception as e:
            return {
                "success": False,
                "error": f"Error truncating {field}: {e}",
                "data": data
            }
        
        return data
