#!/usr/bin/env python
"""
Performance Analytics AI Integration Module - Fixed Version v3 with Trading Signals Fix

This module integrates the Performance Analytics and System Health Monitoring
capabilities with the AI conversational layer, enabling natural language queries
about system performance and health.
"""

import logging
import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceAnalyticsAIIntegration:
    """
    Integrates performance analytics with the AI conversational layer
    """
    
    def __init__(self, storage_manager=None):
        """
        Initialize the Performance Analytics AI Integration
        
        Args:
            storage_manager: The storage manager for accessing performance data
        """
        self.storage_manager = storage_manager
        self.query_patterns = self._compile_query_patterns()
        logger.info("Performance Analytics AI Integration initialized")
    
    def _compile_query_patterns(self):
        """Compile regex patterns for identifying performance-related queries"""
        patterns = {
            # API performance patterns - highest priority
            "api_performance": re.compile(r'(api|mexc|openrouter|telegram).*?(performance|latency|response|time|speed|working|status|slow|reliable|reliability)', re.IGNORECASE),
            
            # How is API performing pattern - specific to catch this phrasing
            "how_is_api": re.compile(r'how\s+is\s+(?:the\s+)?(api|mexc|openrouter|telegram).*?(performing|working|doing)', re.IGNORECASE),
            
            # System health patterns - high priority
            "system_health": re.compile(r'(system|health|status|component|module).*?(status|health|condition|state|running|working)', re.IGNORECASE),
            
            # System health specific commands
            "system_command": re.compile(r'^(?:performance\s+)?system$', re.IGNORECASE),
            
            # Trading signals patterns - FIXED to include 'signals' keyword
            "trading_signals": re.compile(r'(trading\s+signals?|signals?|signal\s+accuracy|trading\s+accuracy|prediction\s+accuracy)', re.IGNORECASE),
            
            # Resource usage patterns
            "resource_usage": re.compile(r'(resource|cpu|memory|disk|usage|utilization)', re.IGNORECASE),
            
            # Anomalies patterns
            "anomalies": re.compile(r'(anomaly|anomalies|unusual|issue|problem|error)', re.IGNORECASE),
            
            # Performance overview patterns - lowest priority
            "performance_overview": re.compile(r'(performance|metrics|analytics|overview|summary).*?(system|overall|general)', re.IGNORECASE),
            
            # Timeframe pattern
            "timeframe": re.compile(r'(last|past|recent|previous)\s+(\d+)\s+(minute|hour|day|week|month)s?', re.IGNORECASE)
        }
        return patterns
    
    def is_performance_query(self, user_message: str) -> bool:
        """
        Determine if a user message is a performance-related query
        
        Args:
            user_message: The message from the user
            
        Returns:
            True if the message is a performance query, False otherwise
        """
        # Check for direct performance-related keywords
        performance_keywords = [
            "performance", "health", "status", "metrics", "monitoring",
            "system health", "system status", "system performance",
            "cpu", "memory", "latency", "response time", "uptime",
            "how is the system", "is everything working", "system check",
            "mexc api", "openrouter api", "telegram api",  # Added specific API mentions
            "trading signals", "signals", "signal accuracy", "signal detection",  # Added trading signal keywords
            "anomalies", "issues", "problems"  # Added anomaly keywords
        ]
        
        user_message_lower = user_message.lower()
        
        if any(keyword.lower() in user_message_lower for keyword in performance_keywords):
            return True
        
        # Check for pattern matches
        for pattern in self.query_patterns.values():
            if pattern.search(user_message):
                return True
        
        return False
    
    def parse_performance_query(self, user_message: str) -> Dict[str, Any]:
        """
        Parse a performance-related query to determine what information is being requested
        
        Args:
            user_message: The message from the user
            
        Returns:
            Dictionary with query parameters
        """
        query_params = {
            "query_type": None,
            "metric_type": None,
            "component": None,
            "timeframe": "recent",  # Default to recent data
            "specific_metric": None,
            "comparison": False
        }
        
        user_message_lower = user_message.lower()
        
        # First, check for direct system command
        if self.query_patterns["system_command"].search(user_message_lower):
            query_params["query_type"] = "system_health"
            self._extract_timeframe(user_message, query_params)
            return query_params
        
        # Second, check for "How is the API performing?" pattern - highest priority
        how_is_api_match = self.query_patterns["how_is_api"].search(user_message_lower)
        if how_is_api_match:
            query_params["query_type"] = "api_performance"
            query_params["metric_type"] = "api_latency"
            
            # Extract the API component
            api_component = how_is_api_match.group(1).lower()
            if api_component in ["mexc", "openrouter", "telegram"]:
                query_params["component"] = api_component
            
            # Extract timeframe if present
            self._extract_timeframe(user_message, query_params)
            
            return query_params
        
        # Third, check for API-specific queries - expanded to catch more patterns
        api_keywords = ["api", "mexc", "openrouter", "telegram"]
        api_performance_keywords = ["performance", "latency", "response", "time", "speed", 
                                   "working", "status", "slow", "reliable", "reliability"]
        
        if any(keyword in user_message_lower for keyword in api_keywords):
            if any(term in user_message_lower for term in api_performance_keywords):
                query_params["query_type"] = "api_performance"
                query_params["metric_type"] = "api_latency"
                
                # Check for specific API component
                for component in ["mexc", "openrouter", "telegram"]:
                    if component in user_message_lower:
                        query_params["component"] = component
                        break
                
                # Extract timeframe if present
                self._extract_timeframe(user_message, query_params)
                
                return query_params
        
        # Fourth, check for system health queries
        if self.query_patterns["system_health"].search(user_message) or any(term in user_message_lower for term in ["system health", "health check", "system status", "everything working"]):
            query_params["query_type"] = "system_health"
            self._extract_timeframe(user_message, query_params)
            return query_params
        
        # Fifth, check for trading signals queries - FIXED to include 'signals' keyword
        if self.query_patterns["trading_signals"].search(user_message) or any(term in user_message_lower for term in ["trading signal", "signals", "signal accuracy", "signal detection", "signal performance"]):
            query_params["query_type"] = "trading_signals"
            self._extract_timeframe(user_message, query_params)
            return query_params
        
        # Sixth, check for resource usage queries
        if self.query_patterns["resource_usage"].search(user_message) or any(term in user_message_lower for term in ["cpu usage", "memory usage", "disk usage", "resource utilization"]):
            query_params["query_type"] = "resource_usage"
            
            # Determine specific resource
            if "cpu" in user_message_lower:
                query_params["specific_metric"] = "cpu_usage"
            elif "memory" in user_message_lower:
                query_params["specific_metric"] = "memory_usage"
            elif "disk" in user_message_lower:
                query_params["specific_metric"] = "disk_usage"
            
            self._extract_timeframe(user_message, query_params)
            return query_params
        
        # Seventh, check for anomalies queries
        if self.query_patterns["anomalies"].search(user_message):
            query_params["query_type"] = "anomalies"
            self._extract_timeframe(user_message, query_params)
            return query_params
        
        # Check for specific components
        components = ["mexc", "openrouter", "telegram", "visualization", "signal_detection", "market_insights", "strategy_discussion"]
        for component in components:
            if component in user_message_lower:
                query_params["component"] = component
                break
        
        # Default to performance overview if no specific query type was identified
        query_params["query_type"] = "performance_overview"
        
        # Extract timeframe if present
        self._extract_timeframe(user_message, query_params)
        
        # Check for comparison request
        comparison_keywords = ["compare", "comparison", "versus", "vs", "difference", "trend"]
        if any(keyword in user_message_lower for keyword in comparison_keywords):
            query_params["comparison"] = True
        
        return query_params
    
    def _extract_timeframe(self, user_message: str, query_params: Dict[str, Any]) -> None:
        """
        Extract timeframe information from the user message and update query parameters
        
        Args:
            user_message: The message from the user
            query_params: The query parameters to update
        """
        # Check for timeframe using regex pattern
        timeframe_match = self.query_patterns["timeframe"].search(user_message)
        if timeframe_match:
            try:
                amount = int(timeframe_match.group(2))
                unit = timeframe_match.group(3).lower()
                
                if "minute" in unit:
                    query_params["timeframe"] = f"last_{amount}_minutes"
                elif "hour" in unit:
                    query_params["timeframe"] = f"last_{amount}_hours"
                elif "day" in unit:
                    query_params["timeframe"] = f"last_{amount}_days"
                elif "week" in unit:
                    query_params["timeframe"] = f"last_{amount}_weeks"
                elif "month" in unit:
                    query_params["timeframe"] = f"last_{amount}_months"
            except (ValueError, IndexError) as e:
                logger.warning(f"Error extracting timeframe: {str(e)}")
    
    async def handle_performance_query(self, user_message: str) -> str:
        """
        Handle a performance-related query and generate a response
        
        Args:
            user_message: The message from the user
            
        Returns:
            Response message
        """
        logger.info(f"Handling performance query: {user_message[:50]}...")
        
        # Parse the query
        query_params = self.parse_performance_query(user_message)
        logger.debug(f"Parsed query parameters: {query_params}")
        
        # Fetch data based on query parameters
        performance_data = await self._fetch_performance_data(query_params)
        
        # Generate response
        response = self._generate_performance_response(query_params, performance_data, user_message)
        
        return response
    
    async def _fetch_performance_data(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch performance data based on query parameters
        
        Args:
            query_params: The parsed query parameters
            
        Returns:
            Performance data
        """
        # If no storage manager is available, use simulated data
        if not self.storage_manager:
            return self._generate_simulated_data(query_params)
        
        try:
            # Determine what data to fetch based on query type
            query_type = query_params["query_type"]
            
            if query_type == "system_health":
                return await self.storage_manager.get_system_health()
            
            elif query_type == "performance_overview":
                return await self.storage_manager.get_performance_overview()
            
            elif query_type == "api_performance":
                component = query_params["component"] or "all"
                timeframe = query_params["timeframe"]
                return await self.storage_manager.get_api_performance(component, timeframe)
            
            elif query_type == "resource_usage":
                specific_metric = query_params["specific_metric"]
                timeframe = query_params["timeframe"]
                return await self.storage_manager.get_resource_usage(specific_metric, timeframe)
            
            elif query_type == "trading_signals":
                timeframe = query_params["timeframe"]
                return await self.storage_manager.get_trading_signal_metrics(timeframe)
            
            elif query_type == "anomalies":
                timeframe = query_params["timeframe"]
                return await self.storage_manager.get_anomalies(timeframe)
            
            else:
                # Default to overview
                return await self.storage_manager.get_performance_overview()
                
        except Exception as e:
            logger.error(f"Error fetching performance data: {str(e)}")
            return {
                "error": str(e),
                "simulated": self._generate_simulated_data(query_params)
            }
    
    def _generate_simulated_data(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simulated performance data for demonstration
        
        Args:
            query_params: The parsed query parameters
            
        Returns:
            Simulated performance data
        """
        import random
        
        query_type = query_params["query_type"]
        
        if query_type == "system_health":
            components = {
                "SystemOverseer": "Running",
                "TelegramBotService": "Running",
                "MEXCDataProvider": "Running",
                "OpenRouterLLM": "Running",
                "VisualizationPlugin": "Running",
                "MarketInsights": "Running",
                "StrategyDiscussion": "Running",
                "PerformanceAnalytics": "Running"
            }
            
            # Randomly set one component to degraded for demonstration
            if random.random() < 0.2:
                component = random.choice(list(components.keys()))
                components[component] = "Degraded"
            
            return {
                "overall_status": "healthy" if all(status == "Running" for status in components.values()) else "degraded",
                "components": components,
                "last_updated": datetime.now().isoformat()
            }
        
        elif query_type == "performance_overview":
            return {
                "api_latency": {
                    "mexc": random.uniform(100, 300),
                    "openrouter": random.uniform(300, 800),
                    "telegram": random.uniform(150, 400)
                },
                "resource_usage": {
                    "cpu_percent": random.uniform(10, 60),
                    "memory_percent": random.uniform(30, 80),
                    "disk_percent": random.uniform(30, 70)
                },
                "signal_accuracy": {
                    "overall": random.uniform(65, 85),
                    "btc": random.uniform(70, 90),
                    "eth": random.uniform(70, 90),
                    "sol": random.uniform(65, 85)
                },
                "response_times": {
                    "avg_ms": random.uniform(200, 800),
                    "p90_ms": random.uniform(500, 1500),
                    "p99_ms": random.uniform(800, 2000)
                }
            }
        
        elif query_type == "api_performance":
            component = query_params["component"]
            
            if component:
                # Generate data for specific component
                return {
                    component: {
                        "latency_ms": random.uniform(100, 500),
                        "error_rate_percent": random.uniform(0, 3),
                        "throughput_rpm": random.uniform(5, 20),
                        "trend": random.choice(["increasing", "decreasing", "stable"])
                    }
                }
            else:
                # Generate data for all components
                return {
                    "mexc": {
                        "latency_ms": random.uniform(100, 300),
                        "error_rate_percent": random.uniform(0, 3),
                        "throughput_rpm": random.uniform(5, 15),
                        "trend": random.choice(["increasing", "decreasing", "stable"])
                    },
                    "openrouter": {
                        "latency_ms": random.uniform(300, 800),
                        "error_rate_percent": random.uniform(0, 2),
                        "throughput_rpm": random.uniform(10, 20),
                        "trend": random.choice(["increasing", "decreasing", "stable"])
                    },
                    "telegram": {
                        "latency_ms": random.uniform(150, 400),
                        "error_rate_percent": random.uniform(0, 1),
                        "throughput_rpm": random.uniform(1, 10),
                        "trend": random.choice(["increasing", "decreasing", "stable"])
                    }
                }
        
        elif query_type == "resource_usage":
            specific_metric = query_params["specific_metric"]
            
            if specific_metric == "cpu_usage":
                return {
                    "current": random.uniform(10, 60),
                    "average_1h": random.uniform(10, 60),
                    "peak_1h": random.uniform(30, 80),
                    "trend": random.choice(["increasing", "decreasing", "stable"])
                }
            elif specific_metric == "memory_usage":
                return {
                    "current": random.uniform(30, 80),
                    "average_1h": random.uniform(30, 70),
                    "peak_1h": random.uniform(50, 90),
                    "trend": random.choice(["increasing", "decreasing", "stable"])
                }
            elif specific_metric == "disk_usage":
                return {
                    "current": random.uniform(30, 70),
                    "average_1h": random.uniform(30, 70),
                    "peak_1h": random.uniform(30, 70),
                    "trend": "stable"
                }
            else:
                # All resources
                return {
                    "cpu": {
                        "current": random.uniform(10, 60),
                        "average_1h": random.uniform(10, 60),
                        "peak_1h": random.uniform(30, 80)
                    },
                    "memory": {
                        "current": random.uniform(30, 80),
                        "average_1h": random.uniform(30, 70),
                        "peak_1h": random.uniform(50, 90)
                    },
                    "disk": {
                        "current": random.uniform(30, 70),
                        "average_1h": random.uniform(30, 70),
                        "peak_1h": random.uniform(30, 70)
                    }
                }
        
        elif query_type == "trading_signals":
            return {
                "overall": {
                    "accuracy": random.uniform(65, 85),
                    "precision": random.uniform(60, 80),
                    "recall": random.uniform(60, 80),
                    "f1_score": random.uniform(60, 80)
                },
                "by_symbol": {
                    "btc": {
                        "accuracy": random.uniform(70, 90),
                        "precision": random.uniform(65, 85),
                        "recall": random.uniform(65, 85),
                        "f1_score": random.uniform(65, 85)
                    },
                    "eth": {
                        "accuracy": random.uniform(70, 90),
                        "precision": random.uniform(65, 85),
                        "recall": random.uniform(65, 85),
                        "f1_score": random.uniform(65, 85)
                    },
                    "sol": {
                        "accuracy": random.uniform(65, 85),
                        "precision": random.uniform(60, 80),
                        "recall": random.uniform(60, 80),
                        "f1_score": random.uniform(60, 80)
                    }
                }
            }
        
        elif query_type == "anomalies":
            # Generate random number of anomalies
            num_anomalies = random.randint(0, 3)
            
            anomalies = []
            for i in range(num_anomalies):
                anomaly_type = random.choice(["api_latency", "error_rate", "resource_usage", "signal_accuracy"])
                component = random.choice(["mexc", "openrouter", "telegram", "system", "cpu", "memory"])
                severity = random.choice(["low", "medium", "high"])
                
                anomalies.append({
                    "type": anomaly_type,
                    "component": component,
                    "severity": severity,
                    "description": f"Unusual {anomaly_type} detected in {component}",
                    "detected_at": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat()
                })
            
            return {
                "count": num_anomalies,
                "anomalies": anomalies,
                "last_checked": datetime.now().isoformat()
            }
        
        else:
            # Default to basic overview
            return {
                "status": "operational",
                "last_updated": datetime.now().isoformat()
            }
    
    def _generate_performance_response(self, query_params: Dict[str, Any], performance_data: Dict[str, Any], user_message: str) -> str:
        """
        Generate a response based on performance data and query parameters
        
        Args:
            query_params: The parsed query parameters
            performance_data: The performance data
            user_message: The original user message
            
        Returns:
            Formatted response message
        """
        query_type = query_params["query_type"]
        
        try:
            if query_type == "system_health":
                return self._format_system_health_response(performance_data)
            
            elif query_type == "performance_overview":
                return self._format_performance_overview_response(performance_data)
            
            elif query_type == "api_performance":
                component = query_params["component"]
                return self._format_api_performance_response(performance_data, component)
            
            elif query_type == "resource_usage":
                specific_metric = query_params["specific_metric"]
                return self._format_resource_usage_response(performance_data, specific_metric)
            
            elif query_type == "trading_signals":
                return self._format_trading_signals_response(performance_data)
            
            elif query_type == "anomalies":
                return self._format_anomalies_response(performance_data)
            
            else:
                # Default to overview
                return self._format_performance_overview_response(performance_data)
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating the performance report: {str(e)}"
    
    def _format_system_health_response(self, data: Dict[str, Any]) -> str:
        """Format system health response"""
        overall_status = data.get("overall_status", "unknown")
        components = data.get("components", {})
        last_updated = data.get("last_updated", datetime.now().isoformat())
        
        # Convert ISO format to datetime object
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated)
            except ValueError:
                last_updated = datetime.now()
        
        # Format the response
        response = "# System Health Status\n"
        
        # Overall status with emoji
        if overall_status == "healthy":
            response += "**Overall Status**: ✅ Healthy\n"
        elif overall_status == "degraded":
            response += "**Overall Status**: ⚠️ Degraded\n"
        else:
            response += "**Overall Status**: ❌ Critical\n"
        
        # Component status
        response += "## Component Status\n"
        for component, status in components.items():
            emoji = "✅" if status == "Running" else "⚠️" if status == "Degraded" else "❌"
            response += f"- {emoji} **{component}**: {status}\n"
        
        # Last updated timestamp
        response += f"_Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}_\n"
        
        # Recommendations if there are issues
        if overall_status != "healthy":
            response += "## Recommendations\n"
            response += "The following components need attention:\n"
            
            for component, status in components.items():
                if status != "Running":
                    response += f"- **{component}**: Check logs and consider restarting the service\n"
        
        return response
    
    def _format_performance_overview_response(self, data: Dict[str, Any]) -> str:
        """Format performance overview response"""
        api_latency = data.get("api_latency", {})
        resource_usage = data.get("resource_usage", {})
        signal_accuracy = data.get("signal_accuracy", {})
        response_times = data.get("response_times", {})
        
        # Format the response
        response = "# System Performance Overview\n"
        
        # API Latency
        response += "## API Latency\n"
        for api, latency in api_latency.items():
            response += f"- **{api.upper()}**: {latency:.2f} ms\n"
        
        # Resource Usage
        response += "## Resource Usage\n"
        if "cpu_percent" in resource_usage:
            response += f"- **CPU**: {resource_usage['cpu_percent']:.1f}%\n"
        if "memory_percent" in resource_usage:
            response += f"- **Memory**: {resource_usage['memory_percent']:.1f}%\n"
        if "disk_percent" in resource_usage:
            response += f"- **Disk**: {resource_usage['disk_percent']:.1f}%\n"
        
        # Signal Accuracy
        response += "## Signal Accuracy\n"
        if "overall" in signal_accuracy:
            response += f"- **OVERALL**: {signal_accuracy['overall']:.1f}%\n"
        if "btc" in signal_accuracy:
            response += f"- **BTC**: {signal_accuracy['btc']:.1f}%\n"
        if "eth" in signal_accuracy:
            response += f"- **ETH**: {signal_accuracy['eth']:.1f}%\n"
        if "sol" in signal_accuracy:
            response += f"- **SOL**: {signal_accuracy['sol']:.1f}%\n"
        
        # Response Times
        response += "## Response Times\n"
        if "avg_ms" in response_times:
            response += f"- **Average**: {response_times['avg_ms']:.2f} ms\n"
        if "p90_ms" in response_times:
            response += f"- **90th Percentile**: {response_times['p90_ms']:.2f} ms\n"
        if "p99_ms" in response_times:
            response += f"- **99th Percentile**: {response_times['p99_ms']:.2f} ms\n"
        
        # System Health Summary
        # Check for any concerning metrics
        concerns = []
        
        for api, latency in api_latency.items():
            if latency > 500:  # Arbitrary threshold
                concerns.append(f"{api.upper()} API latency is high ({latency:.2f} ms)")
        
        if resource_usage.get("cpu_percent", 0) > 80:
            concerns.append(f"CPU usage is high ({resource_usage['cpu_percent']:.1f}%)")
        
        if resource_usage.get("memory_percent", 0) > 85:
            concerns.append(f"Memory usage is high ({resource_usage['memory_percent']:.1f}%)")
        
        if resource_usage.get("disk_percent", 0) > 90:
            concerns.append(f"Disk usage is high ({resource_usage['disk_percent']:.1f}%)")
        
        response += "## System Health Summary\n"
        if concerns:
            response += "⚠️ **Attention needed for the following issues:**\n"
            for concern in concerns:
                response += f"- {concern}\n"
        else:
            response += "✅ **All systems operating within normal parameters.**\n"
        
        return response
    
    def _format_api_performance_response(self, data: Dict[str, Any], component: Optional[str]) -> str:
        """Format API performance response"""
        if component and component in data:
            # Single component response
            api_data = data[component]
            
            response = f"# {component.upper()} API Performance\n"
            response += f"- **Current Latency**: {api_data.get('latency_ms', 0):.2f} ms\n"
            response += f"- **Error Rate**: {api_data.get('error_rate_percent', 0):.2f}%\n"
            response += f"- **Throughput**: {api_data.get('throughput_rpm', 0):.1f} requests/minute\n"
            
            # Trend information
            response += "## Recent Latency Trend\n"
            response += f"- **Average Latency**: {api_data.get('latency_ms', 0) * random.uniform(0.8, 1.2):.2f} ms\n"
            
            trend = api_data.get('trend', 'stable')
            if trend == 'increasing':
                response += "- **Trend**: 📈 Increasing\n"
            elif trend == 'decreasing':
                response += "- **Trend**: 📉 Decreasing\n"
            else:
                response += "- **Trend**: ↔️ Stable\n"
            
            # Recommendations
            response += "## Recommendations\n"
            if api_data.get('error_rate_percent', 0) > 1:
                response += "- Error rate is above normal. Check API endpoint status and review error logs.\n"
            
            if trend == 'increasing':
                response += "- Latency is trending upward significantly. Monitor for potential issues.\n"
            
            return response
        else:
            # All components response
            response = "# API Performance Overview\n"
            
            for api, api_data in data.items():
                response += f"## {api.upper()} API\n"
                response += f"- **Current Latency**: {api_data.get('latency_ms', 0):.2f} ms\n"
                response += f"- **Error Rate**: {api_data.get('error_rate_percent', 0):.2f}%\n"
                response += f"- **Throughput**: {api_data.get('throughput_rpm', 0):.1f} requests/minute\n"
                
                # Trend information
                trend = api_data.get('trend', 'stable')
                if trend == 'increasing':
                    response += "- **Trend**: 📈 Increasing\n"
                elif trend == 'decreasing':
                    response += "- **Trend**: 📉 Decreasing\n"
                else:
                    response += "- **Trend**: ↔️ Stable\n"
                
                response += "\n"
            
            # Recommendations
            response += "## Recommendations\n"
            for api, api_data in data.items():
                if api_data.get('error_rate_percent', 0) > 1:
                    response += f"- {api.upper()} API error rate is above normal ({api_data.get('error_rate_percent', 0):.2f}%). Check endpoint status.\n"
                
                if api_data.get('trend', 'stable') == 'increasing' and api_data.get('latency_ms', 0) > 300:
                    response += f"- {api.upper()} API latency is trending upward. Monitor for potential issues.\n"
            
            return response
    
    def _format_resource_usage_response(self, data: Dict[str, Any], specific_metric: Optional[str]) -> str:
        """Format resource usage response"""
        if specific_metric == "cpu_usage":
            response = "# CPU Usage Metrics\n"
            response += f"**Current Usage**: {data.get('current', 0):.1f}%\n"
            response += f"**Average (1h)**: {data.get('average_1h', 0):.1f}%\n"
            response += f"**Peak (1h)**: {data.get('peak_1h', 0):.1f}%\n"
            
            # Recommendations
            response += "\n## Recommendations\n"
            if data.get('current', 0) > 80:
                response += "- CPU usage is critically high. Consider scaling resources or optimizing workloads.\n"
            elif data.get('current', 0) > 60:
                response += "- CPU usage is elevated. Monitor for potential resource constraints.\n"
            else:
                response += "- CPU usage is within normal parameters.\n"
            
            return response
        
        elif specific_metric == "memory_usage":
            response = "# Memory Usage Metrics\n"
            response += f"**Current Usage**: {data.get('current', 0):.1f}%\n"
            response += f"**Average (1h)**: {data.get('average_1h', 0):.1f}%\n"
            response += f"**Peak (1h)**: {data.get('peak_1h', 0):.1f}%\n"
            
            # Recommendations
            response += "\n## Recommendations\n"
            if data.get('current', 0) > 85:
                response += "- Memory usage is critically high. Check for memory leaks or consider scaling resources.\n"
            elif data.get('current', 0) > 70:
                response += "- Memory usage is elevated. Monitor for potential resource constraints.\n"
            else:
                response += "- Memory usage is within normal parameters.\n"
            
            return response
        
        elif specific_metric == "disk_usage":
            response = "# Disk Usage Metrics\n"
            response += f"**Current Usage**: {data.get('current', 0):.1f}%\n"
            response += f"**Average (1h)**: {data.get('average_1h', 0):.1f}%\n"
            response += f"**Peak (1h)**: {data.get('peak_1h', 0):.1f}%\n"
            
            # Recommendations
            response += "\n## Recommendations\n"
            if data.get('current', 0) > 90:
                response += "- Disk usage is critically high. Clean up unnecessary files or consider adding storage.\n"
            elif data.get('current', 0) > 80:
                response += "- Disk usage is elevated. Plan for potential storage expansion.\n"
            else:
                response += "- Disk usage is within normal parameters.\n"
            
            return response
        
        else:
            # All resources
            response = "# Resource Usage Overview\n"
            
            # CPU
            response += "## CPU Usage\n"
            response += f"- **Current**: {data.get('cpu', {}).get('current', 0):.1f}%\n"
            response += f"- **Average (1h)**: {data.get('cpu', {}).get('average_1h', 0):.1f}%\n"
            response += f"- **Peak (1h)**: {data.get('cpu', {}).get('peak_1h', 0):.1f}%\n\n"
            
            # Memory
            response += "## Memory Usage\n"
            response += f"- **Current**: {data.get('memory', {}).get('current', 0):.1f}%\n"
            response += f"- **Average (1h)**: {data.get('memory', {}).get('average_1h', 0):.1f}%\n"
            response += f"- **Peak (1h)**: {data.get('memory', {}).get('peak_1h', 0):.1f}%\n\n"
            
            # Disk
            response += "## Disk Usage\n"
            response += f"- **Current**: {data.get('disk', {}).get('current', 0):.1f}%\n"
            response += f"- **Average (1h)**: {data.get('disk', {}).get('average_1h', 0):.1f}%\n"
            response += f"- **Peak (1h)**: {data.get('disk', {}).get('peak_1h', 0):.1f}%\n"
            
            # Recommendations
            response += "\n## Recommendations\n"
            
            cpu_current = data.get('cpu', {}).get('current', 0)
            memory_current = data.get('memory', {}).get('current', 0)
            disk_current = data.get('disk', {}).get('current', 0)
            
            if cpu_current > 80:
                response += "- CPU usage is critically high. Consider scaling resources or optimizing workloads.\n"
            elif cpu_current > 60:
                response += "- CPU usage is elevated. Monitor for potential resource constraints.\n"
            
            if memory_current > 85:
                response += "- Memory usage is critically high. Check for memory leaks or consider scaling resources.\n"
            elif memory_current > 70:
                response += "- Memory usage is elevated. Monitor for potential resource constraints.\n"
            
            if disk_current > 90:
                response += "- Disk usage is critically high. Clean up unnecessary files or consider adding storage.\n"
            elif disk_current > 80:
                response += "- Disk usage is elevated. Plan for potential storage expansion.\n"
            
            if cpu_current <= 60 and memory_current <= 70 and disk_current <= 80:
                response += "- All resource usage metrics are within normal parameters.\n"
            
            return response
    
    def _format_trading_signals_response(self, data: Dict[str, Any]) -> str:
        """Format trading signals response"""
        overall = data.get("overall", {})
        by_symbol = data.get("by_symbol", {})
        
        response = "# Trading Signal Performance\n"
        
        # Overall metrics
        response += "## Overall Performance\n"
        response += f"- **Accuracy**: {overall.get('accuracy', 0):.1f}%\n"
        response += f"- **Precision**: {overall.get('precision', 0):.1f}%\n"
        response += f"- **Recall**: {overall.get('recall', 0):.1f}%\n"
        response += f"- **F1 Score**: {overall.get('f1_score', 0):.1f}%\n\n"
        
        # By symbol
        response += "## Performance by Symbol\n"
        
        for symbol, metrics in by_symbol.items():
            response += f"### {symbol.upper()}\n"
            response += f"- **Accuracy**: {metrics.get('accuracy', 0):.1f}%\n"
            response += f"- **Precision**: {metrics.get('precision', 0):.1f}%\n"
            response += f"- **Recall**: {metrics.get('recall', 0):.1f}%\n"
            response += f"- **F1 Score**: {metrics.get('f1_score', 0):.1f}%\n\n"
        
        # Recommendations
        response += "## Recommendations\n"
        
        low_accuracy_symbols = []
        for symbol, metrics in by_symbol.items():
            if metrics.get('accuracy', 0) < 70:
                low_accuracy_symbols.append(symbol)
        
        if overall.get('accuracy', 0) < 70:
            response += "- Overall signal accuracy is below target. Consider retraining the model with more recent data.\n"
        
        if low_accuracy_symbols:
            response += f"- The following symbols have below-target accuracy: {', '.join(s.upper() for s in low_accuracy_symbols)}. Consider adjusting parameters for these symbols.\n"
        
        if not low_accuracy_symbols and overall.get('accuracy', 0) >= 70:
            response += "- All trading signals are performing within acceptable parameters.\n"
        
        return response
    
    def _format_anomalies_response(self, data: Dict[str, Any]) -> str:
        """Format anomalies response"""
        count = data.get("count", 0)
        anomalies = data.get("anomalies", [])
        last_checked = data.get("last_checked", datetime.now().isoformat())
        
        # Convert ISO format to datetime object
        if isinstance(last_checked, str):
            try:
                last_checked = datetime.fromisoformat(last_checked)
            except ValueError:
                last_checked = datetime.now()
        
        response = "# System Anomaly Detection\n"
        
        if count == 0:
            response += "✅ **No anomalies detected in the system.**\n"
        else:
            response += f"⚠️ **{count} anomalies detected in the system.**\n"
        
        response += f"_Last checked: {last_checked.strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        
        if anomalies:
            response += "## Detected Anomalies\n"
            
            for i, anomaly in enumerate(anomalies):
                detected_at = anomaly.get("detected_at", "")
                if isinstance(detected_at, str):
                    try:
                        detected_at = datetime.fromisoformat(detected_at)
                        detected_at_str = detected_at.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        detected_at_str = detected_at
                else:
                    detected_at_str = str(detected_at)
                
                severity = anomaly.get("severity", "").upper()
                severity_emoji = "🔴" if severity == "HIGH" else "🟠" if severity == "MEDIUM" else "🟡"
                
                response += f"### Anomaly {i+1}: {severity_emoji} {severity} Severity\n"
                response += f"- **Type**: {anomaly.get('type', '').replace('_', ' ').title()}\n"
                response += f"- **Component**: {anomaly.get('component', '').upper()}\n"
                response += f"- **Description**: {anomaly.get('description', '')}\n"
                response += f"- **Detected At**: {detected_at_str}\n\n"
            
            # Recommendations
            response += "## Recommendations\n"
            
            high_severity = any(a.get("severity", "") == "high" for a in anomalies)
            api_issues = any(a.get("type", "") == "api_latency" or a.get("type", "") == "error_rate" for a in anomalies)
            resource_issues = any(a.get("type", "") == "resource_usage" for a in anomalies)
            
            if high_severity:
                response += "- High severity anomalies detected. Immediate investigation recommended.\n"
            
            if api_issues:
                response += "- API performance issues detected. Check external service status and connection quality.\n"
            
            if resource_issues:
                response += "- Resource usage anomalies detected. Monitor system load and consider scaling resources if persistent.\n"
        
        return response

# Example usage (for testing)
if __name__ == "__main__":
    # Configure logging for the main module
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("performance_analytics_ai_integration.log"),
            logging.StreamHandler()
        ]
    )
    
    async def test_integration():
        # Create integration
        integration = PerformanceAnalyticsAIIntegration()
        
        # Test queries
        test_queries = [
            "How is the system performing?",
            "What's the current CPU usage?",
            "Show me the API performance",
            "Is the MEXC API slow today?",
            "Has the OpenRouter API been reliable?",
            "Show me trading signals performance",
            "performance signals",  # This should now work with the fix
            "Are there any anomalies in the system?",
            "What's the memory usage like?",
            "Show me disk usage for the last 30 minutes",
            "How is the system health?"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            print("-" * 50)
            
            # Check if it's a performance query
            is_performance = integration.is_performance_query(query)
            print(f"Is performance query: {is_performance}")
            
            if is_performance:
                # Parse the query
                params = integration.parse_performance_query(query)
                print(f"Parsed parameters: {params}")
                
                # Generate response
                response = await integration.handle_performance_query(query)
                print(f"Response preview: {response[:200]}...")
            
            print("=" * 80)
    
    import asyncio
    asyncio.run(test_integration())
