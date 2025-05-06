"""
Session Comparison module for the AI Trading Agent.

This module provides tools to compare multiple paper trading sessions.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import io
import base64

from .session_manager import PaperTradingSession, session_manager


class SessionComparison:
    """
    Provides tools to compare multiple paper trading sessions.
    """
    
    @staticmethod
    def compare_performance_metrics(session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance metrics across multiple sessions.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        sessions = []
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            if session and 'performance_metrics' in session.results:
                sessions.append(session)
        
        if not sessions:
            return {"error": "No valid sessions found with performance metrics"}
        
        # Extract performance metrics from each session
        metrics_data = {}
        for session in sessions:
            metrics = session.results.get('performance_metrics', {})
            session_name = f"Session {session.session_id[:8]}"  # Use shortened ID as name
            metrics_data[session_name] = metrics
        
        # Convert to DataFrame for easier comparison
        metrics_df = pd.DataFrame(metrics_data)
        
        # Calculate differences and rankings
        comparison = {
            "metrics_table": metrics_df.to_dict(),
            "best_performer": {},
            "relative_performance": {}
        }
        
        # Determine best performer for each metric
        for metric in metrics_df.index:
            if metric in ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']:
                # Higher is better
                best_session = metrics_df.loc[metric].idxmax()
                best_value = metrics_df.loc[metric, best_session]
                comparison["best_performer"][metric] = {
                    "session": best_session,
                    "value": best_value
                }
            elif metric in ['max_drawdown', 'volatility']:
                # Lower is better (absolute value for drawdown which is negative)
                best_session = metrics_df.loc[metric].abs().idxmin()
                best_value = metrics_df.loc[metric, best_session]
                comparison["best_performer"][metric] = {
                    "session": best_session,
                    "value": best_value
                }
        
        # Calculate relative performance (percentage of best performer)
        for metric in comparison["best_performer"]:
            best_value = comparison["best_performer"][metric]["value"]
            if best_value == 0:
                continue  # Skip division by zero
                
            relative = {}
            for session in metrics_df.columns:
                value = metrics_df.loc[metric, session]
                if metric in ['max_drawdown', 'volatility']:
                    # For metrics where lower is better
                    if value == 0:
                        relative[session] = 100  # Perfect score
                    else:
                        relative[session] = min(100, abs(best_value / value * 100))
                else:
                    # For metrics where higher is better
                    if best_value == 0:
                        relative[session] = 0
                    else:
                        relative[session] = value / best_value * 100
            
            comparison["relative_performance"][metric] = relative
        
        return comparison
    
    @staticmethod
    def compare_equity_curves(session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare equity curves across multiple sessions.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Dictionary with comparison data and base64 encoded chart
        """
        sessions = []
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            if session and 'portfolio_history' in session.results:
                sessions.append(session)
        
        if not sessions:
            return {"error": "No valid sessions found with portfolio history"}
        
        # Extract portfolio history from each session
        portfolio_data = {}
        for session in sessions:
            history = session.results.get('portfolio_history', [])
            if not history:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(history)
            if 'timestamp' not in df.columns or 'total_value' not in df.columns:
                continue
                
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            session_name = f"Session {session.session_id[:8]}"  # Use shortened ID as name
            portfolio_data[session_name] = df['total_value']
        
        if not portfolio_data:
            return {"error": "No valid portfolio history data found"}
        
        # Combine into a single DataFrame
        combined_df = pd.DataFrame(portfolio_data)
        
        # Normalize to starting value of 100 for fair comparison
        normalized_df = combined_df.div(combined_df.iloc[0]) * 100
        
        # Generate chart
        plt.figure(figsize=(10, 6))
        normalized_df.plot(title='Normalized Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Normalized Value')
        plt.grid(True)
        plt.legend()
        
        # Save chart to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Calculate correlation matrix
        correlation = normalized_df.corr().to_dict()
        
        # Calculate final performance
        final_performance = (normalized_df.iloc[-1] - 100).to_dict()
        
        return {
            "normalized_data": normalized_df.reset_index().to_dict('records'),
            "correlation": correlation,
            "final_performance": final_performance,
            "chart_base64": chart_base64
        }
    
    @staticmethod
    def compare_trade_statistics(session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare trade statistics across multiple sessions.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Dictionary with trade statistics comparison
        """
        sessions = []
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            if session and 'trades' in session.results:
                sessions.append(session)
        
        if not sessions:
            return {"error": "No valid sessions found with trade data"}
        
        # Extract trade data from each session
        trade_stats = {}
        for session in sessions:
            trades = session.results.get('trades', [])
            if not trades:
                continue
                
            # Calculate statistics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
            gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate average trade
            avg_trade = sum(trade.get('pnl', 0) for trade in trades) / total_trades if total_trades > 0 else 0
            
            session_name = f"Session {session.session_id[:8]}"  # Use shortened ID as name
            trade_stats[session_name] = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_trade": avg_trade,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss
            }
        
        if not trade_stats:
            return {"error": "No valid trade statistics could be calculated"}
        
        # Convert to DataFrame for easier comparison
        stats_df = pd.DataFrame(trade_stats)
        
        # Determine best performer for each metric
        best_performer = {}
        for metric in stats_df.index:
            if metric in ['total_trades']:
                # Neutral metric, no best performer
                continue
            elif metric in ['win_rate', 'profit_factor', 'avg_trade', 'gross_profit']:
                # Higher is better
                best_session = stats_df.loc[metric].idxmax()
                best_value = stats_df.loc[metric, best_session]
                best_performer[metric] = {
                    "session": best_session,
                    "value": best_value
                }
            elif metric in ['losing_trades', 'gross_loss']:
                # Lower is better
                best_session = stats_df.loc[metric].idxmin()
                best_value = stats_df.loc[metric, best_session]
                best_performer[metric] = {
                    "session": best_session,
                    "value": best_value
                }
        
        return {
            "trade_stats": stats_df.to_dict(),
            "best_performer": best_performer
        }
    
    @staticmethod
    def generate_comparison_report(session_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report for multiple sessions.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Dictionary with complete comparison report
        """
        if not session_ids or len(session_ids) < 2:
            return {"error": "At least two session IDs are required for comparison"}
        
        # Get basic session info
        sessions_info = []
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            if session:
                sessions_info.append({
                    "session_id": session.session_id,
                    "start_time": session.start_time,
                    "end_time": session.end_time,
                    "status": session.status,
                    "duration_minutes": session.duration_minutes,
                    "symbols": session.symbols
                })
        
        # Compare performance metrics
        performance_comparison = SessionComparison.compare_performance_metrics(session_ids)
        
        # Compare equity curves
        equity_comparison = SessionComparison.compare_equity_curves(session_ids)
        
        # Compare trade statistics
        trade_comparison = SessionComparison.compare_trade_statistics(session_ids)
        
        # Combine into a comprehensive report
        report = {
            "sessions_info": sessions_info,
            "performance_comparison": performance_comparison,
            "equity_comparison": equity_comparison,
            "trade_comparison": trade_comparison,
            "generated_at": datetime.now().isoformat()
        }
        
        return report
