#!/usr/bin/env python
"""Transaction Cost Analysis (TCA) Module

This module provides functionality for analyzing the execution performance
of trades, measuring various metrics like implementation shortfall, market impact,
slippage, and timing costs.

TCA metrics help traders understand the quality of their executions and
adjust their strategies to minimize execution costs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import statistics

from src.common.logging import get_logger
from src.models.order import OrderSide

logger = get_logger("execution.tca")


class TransactionCostAnalyzer:
    """Transaction Cost Analyzer for measuring execution performance.
    
    This class calculates various TCA metrics for completed trades, allowing
    traders to assess execution quality and costs.
    """
    
    def __init__(self):
        """Initialize the Transaction Cost Analyzer."""
        pass
    
    def calculate_implementation_shortfall(
        self,
        side: OrderSide,
        decision_price: float,
        execution_price: float,
        quantity: float,
        fees: float = 0.0
    ) -> Dict[str, float]:
        """Calculate implementation shortfall (the difference between decision price and execution price).
        
        Args:
            side: Order side (buy/sell)
            decision_price: Price at the time of trading decision
            execution_price: Actual average execution price
            quantity: Executed quantity
            fees: Trading fees
            
        Returns:
            Dict containing the following:
                - implementation_shortfall: Total shortfall in price units
                - percentage_shortfall: Shortfall as a percentage of decision price
                - shortfall_cost: Total cost of shortfall in quote currency
        """
        # For buys, a higher execution price is worse (positive shortfall)
        # For sells, a lower execution price is worse (positive shortfall)
        if side == OrderSide.BUY:
            price_shortfall = execution_price - decision_price
        else:
            price_shortfall = decision_price - execution_price
        
        percentage_shortfall = (price_shortfall / decision_price) * 100 if decision_price > 0 else 0
        shortfall_cost = price_shortfall * quantity
        
        # Include fees in the total cost
        total_cost = shortfall_cost + fees
        
        return {
            "implementation_shortfall": price_shortfall,
            "percentage_shortfall": percentage_shortfall,
            "shortfall_cost": shortfall_cost,
            "fees": fees,
            "total_cost": total_cost
        }
    
    def calculate_market_impact(
        self,
        side: OrderSide,
        pre_trade_price: float,
        execution_price: float,
        quantity: float,
        market_volume: float
    ) -> Dict[str, float]:
        """Calculate market impact of a trade.
        
        Args:
            side: Order side (buy/sell)
            pre_trade_price: Market price before trade execution
            execution_price: Actual average execution price
            quantity: Executed quantity
            market_volume: Market volume during execution period
            
        Returns:
            Dict containing the following:
                - price_impact: Impact on price in price units
                - percentage_impact: Impact as a percentage of pre-trade price
                - participation_rate: Trade quantity as a percentage of market volume
        """
        # Calculate price impact
        if side == OrderSide.BUY:
            price_impact = execution_price - pre_trade_price
        else:
            price_impact = pre_trade_price - execution_price
        
        percentage_impact = (price_impact / pre_trade_price) * 100 if pre_trade_price > 0 else 0
        
        # Calculate participation rate
        participation_rate = (quantity / market_volume) * 100 if market_volume > 0 else 0
        
        return {
            "price_impact": price_impact,
            "percentage_impact": percentage_impact,
            "participation_rate": participation_rate
        }
    
    def calculate_slippage(
        self,
        side: OrderSide,
        expected_price: float,
        execution_price: float
    ) -> Dict[str, float]:
        """Calculate execution slippage.
        
        Args:
            side: Order side (buy/sell)
            expected_price: Expected execution price (e.g., mid market)
            execution_price: Actual average execution price
            
        Returns:
            Dict containing slippage metrics
        """
        # Calculate slippage
        if side == OrderSide.BUY:
            slippage = execution_price - expected_price
        else:
            slippage = expected_price - execution_price
        
        percentage_slippage = (slippage / expected_price) * 100 if expected_price > 0 else 0
        
        return {
            "slippage": slippage,
            "percentage_slippage": percentage_slippage
        }
    
    def calculate_timing_cost(
        self,
        side: OrderSide,
        arrival_price: float,
        vwap_price: float,
        execution_price: float
    ) -> Dict[str, float]:
        """Calculate timing cost.
        
        Args:
            side: Order side (buy/sell)
            arrival_price: Market price at order arrival time
            vwap_price: Volume-weighted average price during execution period
            execution_price: Actual average execution price
            
        Returns:
            Dict containing timing cost metrics
        """
        # Calculate arrival price performance
        if side == OrderSide.BUY:
            arrival_performance = arrival_price - execution_price
            vwap_performance = vwap_price - execution_price
        else:
            arrival_performance = execution_price - arrival_price
            vwap_performance = execution_price - vwap_price
        
        percentage_arrival = (arrival_performance / arrival_price) * 100 if arrival_price > 0 else 0
        percentage_vwap = (vwap_performance / vwap_price) * 100 if vwap_price > 0 else 0
        
        return {
            "arrival_performance": arrival_performance,
            "percentage_arrival": percentage_arrival,
            "vwap_performance": vwap_performance,
            "percentage_vwap": percentage_vwap
        }
    
    def analyze_execution_quality(
        self,
        order_details: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive execution quality analysis.
        
        Args:
            order_details: Dictionary containing order execution details
            market_data: Dictionary containing market data during execution
            
        Returns:
            Dict containing comprehensive execution analysis metrics
        """
        # Extract order details
        side = order_details.get("side")
        decision_price = order_details.get("decision_price")
        arrival_price = order_details.get("arrival_price")
        expected_price = order_details.get("expected_price")
        execution_price = order_details.get("execution_price")
        quantity = order_details.get("quantity")
        fees = order_details.get("fees", 0.0)
        
        # Extract market data
        market_volume = market_data.get("volume", 0)
        vwap_price = market_data.get("vwap_price")
        pre_trade_price = market_data.get("pre_trade_price")
        
        # Calculate all metrics
        result = {
            "order_details": order_details,
            "execution_time": order_details.get("execution_time"),
            "quantity": quantity,
            "execution_price": execution_price,
        }
        
        # Implementation shortfall
        if (side is not None and isinstance(side, OrderSide) and 
            decision_price is not None and execution_price is not None and 
            quantity is not None):
            result["implementation_shortfall"] = self.calculate_implementation_shortfall(
                side=side,
                decision_price=float(decision_price),
                execution_price=float(execution_price),
                quantity=float(quantity),
                fees=float(fees)
            )
        
        # Market impact
        if (side is not None and isinstance(side, OrderSide) and 
            pre_trade_price is not None and execution_price is not None and
            quantity is not None):
            result["market_impact"] = self.calculate_market_impact(
                side=side,
                pre_trade_price=float(pre_trade_price),
                execution_price=float(execution_price),
                quantity=float(quantity),
                market_volume=float(market_volume)
            )
        
        # Slippage
        if (side is not None and isinstance(side, OrderSide) and 
            expected_price is not None and execution_price is not None):
            result["slippage"] = self.calculate_slippage(
                side=side,
                expected_price=float(expected_price),
                execution_price=float(execution_price)
            )
        
        # Timing cost
        if (side is not None and isinstance(side, OrderSide) and 
            arrival_price is not None and vwap_price is not None and 
            execution_price is not None):
            result["timing_cost"] = self.calculate_timing_cost(
                side=side,
                arrival_price=float(arrival_price),
                vwap_price=float(vwap_price),
                execution_price=float(execution_price)
            )
        
        return result
    
    def analyze_algo_performance(
        self,
        algo_name: str,
        orders: List[Dict[str, Any]],
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the performance of an execution algorithm over multiple orders.
        
        Args:
            algo_name: Name of the execution algorithm
            orders: List of order detail dictionaries
            market_data: Dictionary mapping order IDs to market data
            
        Returns:
            Dict containing aggregated performance metrics
        """
        if not orders:
            return {
                "algo_name": algo_name,
                "order_count": 0,
                "total_volume": 0,
                "average_metrics": {},
                "orders": []
            }
        
        # Analyze each order
        order_analyses = []
        for order in orders:
            order_id = order.get("order_id")
            if order_id is not None and order_id in market_data:
                analysis = self.analyze_execution_quality(order, market_data[order_id])
                order_analyses.append(analysis)
        
        # Calculate aggregated metrics
        total_volume = sum(analysis.get("quantity", 0) for analysis in order_analyses)
        
        # Calculate volume-weighted averages for key metrics
        metrics = {}
        
        # Implementation shortfall
        shortfall_values = [
            analysis.get("implementation_shortfall", {}).get("percentage_shortfall")
            for analysis in order_analyses
            if "implementation_shortfall" in analysis
        ]
        if shortfall_values:
            metrics["avg_percentage_shortfall"] = sum(shortfall_values) / len(shortfall_values)
            metrics["max_percentage_shortfall"] = max(shortfall_values)
            metrics["min_percentage_shortfall"] = min(shortfall_values)
        
        # Market impact
        impact_values = [
            analysis.get("market_impact", {}).get("percentage_impact")
            for analysis in order_analyses
            if "market_impact" in analysis
        ]
        if impact_values:
            metrics["avg_percentage_impact"] = sum(impact_values) / len(impact_values)
            metrics["max_percentage_impact"] = max(impact_values)
            metrics["min_percentage_impact"] = min(impact_values)
        
        # Slippage
        slippage_values = [
            analysis.get("slippage", {}).get("percentage_slippage")
            for analysis in order_analyses
            if "slippage" in analysis
        ]
        if slippage_values:
            metrics["avg_percentage_slippage"] = sum(slippage_values) / len(slippage_values)
            metrics["max_percentage_slippage"] = max(slippage_values)
            metrics["min_percentage_slippage"] = min(slippage_values)
        
        return {
            "algo_name": algo_name,
            "order_count": len(orders),
            "total_volume": total_volume,
            "average_metrics": metrics,
            "orders": order_analyses
        }
    
    def compare_algos(
        self,
        algo_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare the performance of different execution algorithms.
        
        Args:
            algo_results: Dictionary mapping algorithm names to their performance results
            
        Returns:
            Dict containing comparison metrics and rankings
        """
        if not algo_results:
            return {"comparison": "No algorithms to compare"}
        
        # Extract metrics for comparison
        metrics_to_compare = [
            "avg_percentage_shortfall",
            "avg_percentage_impact",
            "avg_percentage_slippage"
        ]
        
        # Create rankings for each metric (lower is better)
        rankings = {metric: [] for metric in metrics_to_compare}
        
        for algo_name, result in algo_results.items():
            for metric in metrics_to_compare:
                value = result.get("average_metrics", {}).get(metric)
                if value is not None:
                    rankings[metric].append((algo_name, value))
        
        # Sort rankings (lower values are better for these metrics)
        for metric in rankings:
            rankings[metric].sort(key=lambda x: x[1])
        
        # Determine overall best algorithm based on average rank
        algo_avg_ranks = {}
        for algo_name in algo_results:
            ranks = []
            for metric in rankings:
                for i, (name, _) in enumerate(rankings[metric]):
                    if name == algo_name:
                        ranks.append(i + 1)  # 1-based ranking
            
            if ranks:
                algo_avg_ranks[algo_name] = sum(ranks) / len(ranks)
        
        # Sort by average rank
        overall_ranking = sorted(algo_avg_ranks.items(), key=lambda x: x[1])
        
        return {
            "metric_rankings": rankings,
            "overall_ranking": overall_ranking,
            "best_algorithm": overall_ranking[0][0] if overall_ranking else None,
            "detailed_results": algo_results
        }


class RealTimeMetrics:
    """Real-time execution metrics calculator.
    
    This class maintains a running calculation of execution metrics that
    can be updated in real-time as orders are executed.
    """
    
    def __init__(self, side: OrderSide, decision_price: float, expected_quantity: float):
        """Initialize the real-time metrics calculator.
        
        Args:
            side: Order side (buy/sell)
            decision_price: Price at the time of trading decision
            expected_quantity: Total expected execution quantity
        """
        self.side = side
        self.decision_price = decision_price
        self.expected_quantity = expected_quantity
        
        # Tracking variables
        self.executed_quantity = 0.0
        self.total_value = 0.0
        self.last_update_time = datetime.now()
        self.start_time = datetime.now()
        self.fill_times = []
        self.fill_prices = []
        self.fill_quantities = []
        
    def update(self, price: float, quantity: float, timestamp: Optional[datetime] = None):
        """Update metrics with a new fill.
        
        Args:
            price: Fill price
            quantity: Fill quantity
            timestamp: Optional fill timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.executed_quantity += quantity
        self.total_value += price * quantity
        self.last_update_time = timestamp
        
        self.fill_times.append(timestamp)
        self.fill_prices.append(price)
        self.fill_quantities.append(quantity)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current execution metrics.
        
        Returns:
            Dict containing current execution metrics
        """
        # Calculate current average price
        avg_price = self.total_value / self.executed_quantity if self.executed_quantity > 0 else 0
        
        # Calculate implementation shortfall
        if self.side == OrderSide.BUY:
            price_shortfall = avg_price - self.decision_price
        else:
            price_shortfall = self.decision_price - avg_price
        
        percentage_shortfall = (price_shortfall / self.decision_price) * 100 if self.decision_price > 0 else 0
        total_shortfall_cost = price_shortfall * self.executed_quantity
        
        # Calculate completion percentage
        completion_pct = (self.executed_quantity / self.expected_quantity) * 100 if self.expected_quantity > 0 else 0
        
        # Calculate execution speed
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        execution_rate = self.executed_quantity / elapsed_seconds if elapsed_seconds > 0 else 0
        
        # Calculate price dispersion (standard deviation of fill prices)
        price_std_dev = statistics.stdev(self.fill_prices) if len(self.fill_prices) > 1 else 0
        
        return {
            "executed_quantity": self.executed_quantity,
            "average_price": avg_price,
            "completion_percentage": completion_pct,
            "implementation_shortfall": price_shortfall,
            "percentage_shortfall": percentage_shortfall,
            "total_shortfall_cost": total_shortfall_cost,
            "execution_time_seconds": elapsed_seconds,
            "execution_rate": execution_rate,
            "price_dispersion": price_std_dev,
            "remaining_quantity": self.expected_quantity - self.executed_quantity
        } 