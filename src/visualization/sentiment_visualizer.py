"""Visualization tools for sentiment analysis data.

This module provides tools for visualizing sentiment data and its
relationship with price movements and trading signals.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import networkx as nx

class SentimentVisualizer:
    """Visualization tools for sentiment analysis.
    
    This class provides methods for creating visualizations of sentiment
    data, including time series, correlation with price, and network
    visualizations of connected events.
    """
    
    def __init__(self):
        """Initialize the sentiment visualizer."""
        self.style_setup()
    
    def style_setup(self) -> None:
        """Set up visualization style."""
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
    
    def plot_sentiment_price(self,
                             sentiment_data: pd.DataFrame,
                             price_data: pd.DataFrame,
                             title: str = "Sentiment and Price Relationship",
                             save_path: Optional[str] = None) -> None:
        """Plot sentiment data alongside price data.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            price_data: DataFrame with price data
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Ensure DataFrames have datetime index
        sentiment_data = sentiment_data.copy()
        price_data = price_data.copy()
        
        if "timestamp" in sentiment_data.columns:
            sentiment_data["timestamp"] = pd.to_datetime(sentiment_data["timestamp"])
            sentiment_data.set_index("timestamp", inplace=True)
            
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data.set_index("timestamp", inplace=True)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Price chart
        ax1 = plt.subplot(gs[0])
        ax1.plot(price_data.index, price_data["close"], color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Price", fontweight="bold")
        ax1.set_title(title, fontsize=16, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Sentiment chart
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(sentiment_data.index, sentiment_data["value"], color="#ff7f0e", linewidth=2)
        ax2.set_ylabel("Sentiment", fontweight="bold")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add neutral line
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        
        # Add bands for bullish/bearish sentiment
        ax2.axhspan(0.7, 1.0, alpha=0.2, color="green", label="Bullish")
        ax2.axhspan(0.0, 0.3, alpha=0.2, color="red", label="Bearish")
        ax2.legend(loc="upper right")
        
        # Confidence chart
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(sentiment_data.index, sentiment_data["confidence"], color="#2ca02c", linewidth=2)
        ax3.set_ylabel("Confidence", fontweight="bold")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add threshold line
        ax3.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Threshold")
        ax3.legend(loc="upper right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    def plot_sentiment_sources(self,
                               sentiment_data: Dict[str, pd.DataFrame],
                               price_data: pd.DataFrame,
                               title: str = "Sentiment by Source",
                               save_path: Optional[str] = None) -> None:
        """Plot sentiment data from multiple sources.
        
        Args:
            sentiment_data: Dictionary of DataFrames with sentiment data by source
            price_data: DataFrame with price data
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Ensure price DataFrame has datetime index
        price_data = price_data.copy()
        
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data.set_index("timestamp", inplace=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 6 + 3 * len(sentiment_data)))
        rows = len(sentiment_data) + 1
        gs = gridspec.GridSpec(rows, 1, height_ratios=[2] + [1] * len(sentiment_data))
        
        # Price chart
        ax1 = plt.subplot(gs[0])
        ax1.plot(price_data.index, price_data["close"], color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Price", fontweight="bold")
        ax1.set_title(title, fontsize=16, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Color map for sources
        colors = plt.cm.tab10(np.linspace(0, 1, len(sentiment_data)))
        
        # Plot each sentiment source
        for i, (source, data) in enumerate(sentiment_data.items(), 1):
            # Ensure DataFrame has datetime index
            data = data.copy()
            
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data.set_index("timestamp", inplace=True)
                
            # Create subplot
            ax = plt.subplot(gs[i], sharex=ax1)
            ax.plot(data.index, data["value"], color=colors[i-1], linewidth=2)
            ax.set_ylabel(f"{source}\nSentiment", fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add neutral line
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            
            # Add confidence as area
            if "confidence" in data.columns:
                ax.fill_between(data.index, 0, data["value"], 
                               alpha=0.3, color=colors[i-1])
                
            # Add source label
            ax.text(0.02, 0.9, source, transform=ax.transAxes,
                   fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    def plot_sentiment_signals(self,
                               sentiment_data: pd.DataFrame,
                               price_data: pd.DataFrame,
                               signals: pd.DataFrame,
                               title: str = "Sentiment Signals and Performance",
                               save_path: Optional[str] = None) -> None:
        """Plot sentiment data with trading signals.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            price_data: DataFrame with price data
            signals: DataFrame with signal data (timestamp, direction)
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Ensure DataFrames have datetime index
        sentiment_data = sentiment_data.copy()
        price_data = price_data.copy()
        signals = signals.copy()
        
        if "timestamp" in sentiment_data.columns:
            sentiment_data["timestamp"] = pd.to_datetime(sentiment_data["timestamp"])
            sentiment_data.set_index("timestamp", inplace=True)
            
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data.set_index("timestamp", inplace=True)
            
        if "timestamp" in signals.columns:
            signals["timestamp"] = pd.to_datetime(signals["timestamp"])
            signals.set_index("timestamp", inplace=True)
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        # Price chart with signals
        ax1 = plt.subplot(gs[0])
        ax1.plot(price_data.index, price_data["close"], color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Price", fontweight="bold")
        ax1.set_title(title, fontsize=16, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        # Add signals to price chart
        for idx, row in signals.iterrows():
            if row["direction"] == "buy":
                ax1.scatter(idx, price_data.loc[idx, "close"], 
                           color="green", marker="^", s=100, zorder=5)
            elif row["direction"] == "sell":
                ax1.scatter(idx, price_data.loc[idx, "close"], 
                           color="red", marker="v", s=100, zorder=5)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Sentiment chart
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(sentiment_data.index, sentiment_data["value"], color="#ff7f0e", linewidth=2)
        ax2.set_ylabel("Sentiment", fontweight="bold")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add neutral line
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        
        # Add bands for bullish/bearish sentiment
        ax2.axhspan(0.7, 1.0, alpha=0.2, color="green", label="Bullish")
        ax2.axhspan(0.0, 0.3, alpha=0.2, color="red", label="Bearish")
        
        # Add signals to sentiment chart
        for idx, row in signals.iterrows():
            sentiment_value = sentiment_data.loc[sentiment_data.index <= idx, "value"].iloc[-1]
            if row["direction"] == "buy":
                ax2.scatter(idx, sentiment_value, 
                           color="green", marker="^", s=100, zorder=5)
            elif row["direction"] == "sell":
                ax2.scatter(idx, sentiment_value, 
                           color="red", marker="v", s=100, zorder=5)
        
        ax2.legend(loc="upper right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    def plot_sentiment_correlation(self,
                                   sentiment_data: pd.DataFrame,
                                   price_data: pd.DataFrame,
                                   title: str = "Sentiment-Price Correlation",
                                   save_path: Optional[str] = None) -> None:
        """Plot correlation between sentiment and price changes.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            price_data: DataFrame with price data
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Ensure DataFrames have datetime index
        sentiment_data = sentiment_data.copy()
        price_data = price_data.copy()
        
        if "timestamp" in sentiment_data.columns:
            sentiment_data["timestamp"] = pd.to_datetime(sentiment_data["timestamp"])
            sentiment_data.set_index("timestamp", inplace=True)
            
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data.set_index("timestamp", inplace=True)
        
        # Calculate price changes
        price_data["price_change"] = price_data["close"].pct_change(24).shift(-24)  # 24h forward return
        
        # Merge data
        merged_data = pd.merge(
            sentiment_data[["value", "confidence"]],
            price_data[["close", "price_change"]],
            left_index=True,
            right_index=True,
            how="inner"
        )
        
        # Drop rows with NaN
        merged_data = merged_data.dropna()
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            merged_data["value"],
            merged_data["price_change"] * 100,  # Convert to percentage
            c=merged_data["confidence"],
            cmap="viridis",
            alpha=0.7,
            s=50
        )
        
        # Add colorbar for confidence
        cbar = plt.colorbar(scatter)
        cbar.set_label("Confidence", rotation=270, labelpad=15, fontweight="bold")
        
        # Add regression line
        z = np.polyfit(merged_data["value"], merged_data["price_change"] * 100, 1)
        p = np.poly1d(z)
        ax.plot(
            merged_data["value"],
            p(merged_data["value"]),
            "r--",
            alpha=0.8,
            linewidth=2
        )
        
        # Calculate correlation
        correlation = merged_data["value"].corr(merged_data["price_change"])
        
        # Add labels and title
        ax.set_xlabel("Sentiment Value", fontweight="bold")
        ax.set_ylabel("24h Price Change (%)", fontweight="bold")
        ax.set_title(f"{title}\nCorrelation: {correlation:.2f}", fontsize=16, fontweight="bold")
        
        # Add vertical line at neutral sentiment
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        
        # Add horizontal line at zero price change
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    def plot_sentiment_extremes(self,
                               sentiment_data: pd.DataFrame,
                               price_data: pd.DataFrame,
                               threshold: float = 0.8,
                               title: str = "Extreme Sentiment Analysis",
                               save_path: Optional[str] = None) -> None:
        """Plot extreme sentiment values and their relationship with price reversals.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            price_data: DataFrame with price data
            threshold: Threshold for extreme sentiment (0-1)
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Ensure DataFrames have datetime index
        sentiment_data = sentiment_data.copy()
        price_data = price_data.copy()
        
        if "timestamp" in sentiment_data.columns:
            sentiment_data["timestamp"] = pd.to_datetime(sentiment_data["timestamp"])
            sentiment_data.set_index("timestamp", inplace=True)
            
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data.set_index("timestamp", inplace=True)
        
        # Find extreme sentiment values
        extreme_bullish = sentiment_data[sentiment_data["value"] >= threshold].index
        extreme_bearish = sentiment_data[sentiment_data["value"] <= (1 - threshold)].index
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax.plot(price_data.index, price_data["close"], color="#1f77b4", linewidth=2)
        ax.set_ylabel("Price", fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        
        # Mark extreme sentiment points
        for idx in extreme_bullish:
            if idx in price_data.index:
                ax.axvline(x=idx, color="green", alpha=0.3, ymax=0.9)
                ax.scatter(idx, price_data.loc[idx, "close"], 
                          color="green", marker="o", s=80, zorder=5)
                
        for idx in extreme_bearish:
            if idx in price_data.index:
                ax.axvline(x=idx, color="red", alpha=0.3, ymax=0.9)
                ax.scatter(idx, price_data.loc[idx, "close"], 
                          color="red", marker="o", s=80, zorder=5)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", 
                  markersize=10, label="Extreme Bullish"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", 
                  markersize=10, label="Extreme Bearish")
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    def plot_event_network(self,
                          event_graph: nx.Graph,
                          title: str = "Event Relationship Network",
                          save_path: Optional[str] = None) -> None:
        """Plot network visualization of connected events.
        
        Args:
            event_graph: NetworkX graph of events and relationships
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Get node attributes
        node_types = nx.get_node_attributes(event_graph, "type")
        node_importance = nx.get_node_attributes(event_graph, "importance")
        
        # Set default importance if not provided
        for node in event_graph.nodes():
            if node not in node_importance:
                node_importance[node] = 0.5
        
        # Set node colors based on type
        node_colors = []
        color_map = {
            "news": "#1f77b4",
            "social_media": "#ff7f0e",
            "geopolitical": "#2ca02c",
            "market": "#d62728",
            "onchain": "#9467bd"
        }
        
        for node in event_graph.nodes():
            node_type = node_types.get(node, "unknown")
            node_colors.append(color_map.get(node_type, "gray"))
        
        # Set node sizes based on importance
        node_sizes = [1000 * node_importance.get(node, 0.5) for node in event_graph.nodes()]
        
        # Get edge attributes
        edge_weights = nx.get_edge_attributes(event_graph, "strength")
        
        # Set default weight if not provided
        for edge in event_graph.edges():
            if edge not in edge_weights:
                edge_weights[edge] = 0.5
        
        # Set edge widths based on strength
        edge_widths = [3 * edge_weights.get(edge, 0.5) for edge in event_graph.edges()]
        
        # Calculate layout
        pos = nx.spring_layout(event_graph, k=0.3, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(
            event_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            event_graph, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color="gray"
        )
        
        nx.draw_networkx_labels(
            event_graph, pos,
            font_size=10,
            font_family="sans-serif"
        )
        
        # Add title
        plt.title(title, fontsize=16, fontweight="bold")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = []
        
        for node_type, color in color_map.items():
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color, 
                      markersize=10, label=node_type.capitalize())
            )
            
        ax.legend(handles=legend_elements, loc="upper right")
        
        # Remove axis
        plt.axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    def plot_causal_chain(self,
                         causal_chain: Dict[str, Any],
                         price_data: Optional[pd.DataFrame] = None,
                         title: str = "Causal Chain Analysis",
                         save_path: Optional[str] = None) -> None:
        """Plot visualization of a causal chain of events.
        
        Args:
            causal_chain: Dictionary with causal chain data
            price_data: Optional DataFrame with price data
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Get events from causal chain
        events = causal_chain.get("events", [])
        
        if not events:
            print("No events in causal chain")
            return
        
        # Create figure
        fig = plt.figure(figsize=(14, 8))
        
        # Calculate grid layout based on whether price data is included
        if price_data is not None:
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
        else:
            ax2 = plt.subplot(111)
        
        # Plot price data if provided
        if price_data is not None:
            # Ensure DataFrame has datetime index
            price_data = price_data.copy()
            
            if "timestamp" in price_data.columns:
                price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
                price_data.set_index("timestamp", inplace=True)
            
            # Plot price
            ax1.plot(price_data.index, price_data["close"], color="#1f77b4", linewidth=2)
            ax1.set_ylabel("Price", fontweight="bold")
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Mark event timestamps on price chart
            for event in events:
                if "timestamp" in event:
                    timestamp = pd.to_datetime(event["timestamp"])
                    if timestamp in price_data.index:
                        ax1.axvline(x=timestamp, color="green", alpha=0.3)
        
        # Plot causal chain as directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, event in enumerate(events):
            G.add_node(i, 
                      description=event.get("description", f"Event {i}"),
                      importance=event.get("importance", 0.5),
                      type=event.get("source", "unknown"))
        
        # Add edges
        for i in range(len(events) - 1):
            G.add_edge(i, i + 1, strength=causal_chain.get("strength", 0.5))
        
        # Calculate position - use hierarchical layout for causality
        pos = nx.spring_layout(G, k=0.5, iterations=100)
        
        # Get node attributes
        node_types = nx.get_node_attributes(G, "type")
        node_importance = nx.get_node_attributes(G, "importance")
        node_descriptions = nx.get_node_attributes(G, "description")
        
        # Set node colors based on type
        node_colors = []
        color_map = {
            "news": "#1f77b4",
            "social_media": "#ff7f0e",
            "geopolitical": "#2ca02c",
            "market": "#d62728",
            "onchain": "#9467bd"
        }
        
        for node in G.nodes():
            node_type = node_types.get(node, "unknown")
            node_colors.append(color_map.get(node_type, "gray"))
        
        # Set node sizes based on importance
        node_sizes = [1000 * node_importance.get(node, 0.5) for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax2
        )
        
        nx.draw_networkx_edges(
            G, pos,
            width=2,
            alpha=0.6,
            edge_color="gray",
            arrowstyle="->",
            arrowsize=20,
            ax=ax2
        )
        
        # Create custom labels with truncated descriptions
        labels = {}
        for node, desc in node_descriptions.items():
            if len(desc) > 30:
                labels[node] = desc[:27] + "..."
            else:
                labels[node] = desc
        
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_family="sans-serif",
            ax=ax2
        )
        
        # Add impact info
        impact_direction = causal_chain.get("impact_direction", "unknown")
        impact_confidence = causal_chain.get("confidence", 0.0)
        
        impact_text = f"Impact Direction: {impact_direction.capitalize()}\n"
        impact_text += f"Confidence: {impact_confidence:.2f}\n"
        impact_text += f"Strength: {causal_chain.get('strength', 0.0):.2f}"
        
        ax2.text(0.02, 0.02, impact_text,
               transform=ax2.transAxes,
               fontsize=12,
               bbox=dict(facecolor="white", alpha=0.8))
        
        # Add title
        if price_data is not None:
            fig.suptitle(title, fontsize=16, fontweight="bold")
        else:
            ax2.set_title(title, fontsize=16, fontweight="bold")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = []
        
        for node_type, color in color_map.items():
            if node_type in node_types.values():
                legend_elements.append(
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=color, 
                          markersize=10, label=node_type.capitalize())
                )
                
        ax2.legend(handles=legend_elements, loc="upper right")
        
        # Remove axis
        ax2.axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample data generation
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
    
    # Sample sentiment data
    sentiment_data = pd.DataFrame({
        "timestamp": dates,
        "value": np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + 0.1 * np.random.randn(len(dates)), 0, 1),
        "confidence": np.clip(0.7 + 0.2 * np.random.randn(len(dates)), 0, 1)
    })
    
    # Sample price data
    base_price = 20000
    noise = np.random.randn(len(dates))
    trend = np.linspace(0, 5000, len(dates))
    cycle = 2000 * np.sin(np.linspace(0, 3*np.pi, len(dates)))
    
    price_data = pd.DataFrame({
        "timestamp": dates,
        "close": base_price + trend + cycle + 200 * noise,
        "open": base_price + trend + cycle + 200 * np.roll(noise, 1),
        "high": base_price + trend + cycle + 400 * np.random.rand(len(dates)),
        "low": base_price + trend + cycle - 400 * np.random.rand(len(dates)),
        "volume": 1000 + 500 * np.abs(np.random.randn(len(dates)))
    })
    
    # Sample signals
    signal_indices = [3, 7, 15, 22, 28]
    signals = pd.DataFrame({
        "timestamp": [dates[i] for i in signal_indices],
        "direction": ["buy", "sell", "buy", "sell", "buy"]
    })
    
    # Initialize visualizer
    visualizer = SentimentVisualizer()
    
    # Plot sentiment and price
    print("Plotting sentiment and price chart...")
    visualizer.plot_sentiment_price(sentiment_data, price_data)
    
    # Plot sentiment signals
    print("Plotting sentiment signals chart...")
    visualizer.plot_sentiment_signals(sentiment_data, price_data, signals)
    
    # Plot sentiment correlation
    print("Plotting sentiment correlation chart...")
    visualizer.plot_sentiment_correlation(sentiment_data, price_data)
    
    # Plot sentiment extremes
    print("Plotting sentiment extremes chart...")
    visualizer.plot_sentiment_extremes(sentiment_data, price_data)
    
    # Create sample event network
    G = nx.Graph()
    
    # Add nodes
    G.add_node(1, type="news", importance=0.8, description="Major economic announcement")
    G.add_node(2, type="social_media", importance=0.6, description="Trending crypto topic")
    G.add_node(3, type="geopolitical", importance=0.9, description="Regulatory change")
    G.add_node(4, type="market", importance=0.7, description="Market volatility increase")
    G.add_node(5, type="onchain", importance=0.5, description="Large wallet movement")
    
    # Add edges
    G.add_edge(1, 3, strength=0.9)
    G.add_edge(1, 2, strength=0.7)
    G.add_edge(2, 4, strength=0.6)
    G.add_edge(3, 4, strength=0.8)
    G.add_edge(3, 5, strength=0.5)
    G.add_edge(4, 5, strength=0.4)
    
    # Plot event network
    print("Plotting event network...")
    visualizer.plot_event_network(G)
    
    # Create sample causal chain
    causal_chain = {
        "id": "chain-001",
        "events": [
            {
                "id": "event-001",
                "description": "Central bank announces policy change",
                "source": "news",
                "importance": 0.9,
                "timestamp": "2023-01-05"
            },
            {
                "id": "event-002",
                "description": "Market participants react on social media",
                "source": "social_media",
                "importance": 0.7,
                "timestamp": "2023-01-06"
            },
            {
                "id": "event-003",
                "description": "Institutional investors reposition",
                "source": "market",
                "importance": 0.8,
                "timestamp": "2023-01-08"
            },
            {
                "id": "event-004",
                "description": "Large outflows from exchanges",
                "source": "onchain",
                "importance": 0.6,
                "timestamp": "2023-01-10"
            }
        ],
        "strength": 0.85,
        "market_relevance": 0.9,
        "target_assets": ["BTC/USDT"],
        "impact_direction": "positive",
        "confidence": 0.8
    }
    
    # Plot causal chain
    print("Plotting causal chain...")
    visualizer.plot_causal_chain(causal_chain, price_data)
    
    print("All visualizations completed!")