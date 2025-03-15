"""Visualization tools for market regime analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray

class RegimeVisualizer:
    """Visualize market regime analysis results."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.colors = {
            0: 'red',    # Negative/Low/Bearish
            1: 'gray',   # Neutral/Normal
            2: 'green'   # Positive/High/Bullish
        }
        
    def plot_regime_transitions(
        self,
        dates: NDArray,
        labels: NDArray[np.int64],
        prices: Optional[NDArray[np.float64]] = None,
        regime_names: Optional[List[str]] = None,
        title: str = "Market Regime Transitions"
    ) -> go.Figure:
        """Plot regime transitions with optional price overlay.
        
        Args:
            dates: Array of dates
            labels: Array of regime labels
            prices: Optional array of prices for overlay
            regime_names: Optional list of regime names
            title: Plot title
        
        Returns:
            Plotly figure
        """
        if regime_names is None:
            regime_names = ['Regime 0', 'Regime 1', 'Regime 2']
            
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Plot price if provided
        if prices is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prices,
                    name='Price',
                    line=dict(color='black', width=1)
                ),
                row=1, col=1
            )
        
        # Plot regime background colors
        for regime in range(3):
            mask = labels == regime
            if not any(mask):
                continue
                
            fig.add_trace(
                go.Scatter(
                    x=dates[mask],
                    y=[regime] * sum(mask),
                    name=regime_names[regime],
                    mode='markers',
                    marker=dict(
                        color=self.colors[regime],
                        size=10
                    )
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800
        )
        
        return fig
    
    def plot_regime_statistics(
        self,
        stats: Dict[int, Dict[str, float]],
        regime_names: Optional[List[str]] = None
    ) -> go.Figure:
        """Plot regime statistics comparison.
        
        Args:
            stats: Dictionary of regime statistics
            regime_names: Optional list of regime names
        
        Returns:
            Plotly figure
        """
        if regime_names is None:
            regime_names = ['Regime 0', 'Regime 1', 'Regime 2']
            
        metrics = [
            'mean_return', 'volatility', 'sharpe_ratio',
            'skewness', 'kurtosis', 'var_95', 'frequency'
        ]
        
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.05
        )
        
        for metric_idx, metric in enumerate(metrics, 1):
            values = [stats[r][metric] for r in range(len(regime_names))]
            
            fig.add_trace(
                go.Bar(
                    x=regime_names,
                    y=values,
                    marker_color=[self.colors[i] for i in range(len(regime_names))]
                ),
                row=metric_idx, col=1
            )
        
        fig.update_layout(
            height=200 * len(metrics),
            showlegend=False,
            title="Regime Statistics Comparison"
        )
        
        return fig
    
    def plot_regime_heatmap(
        self,
        regime_matrix: NDArray[np.int64],
        dates: NDArray,
        method_names: List[str],
        title: str = "Regime Detection Method Comparison"
    ) -> go.Figure:
        """Plot heatmap comparing different regime detection methods.
        
        Args:
            regime_matrix: Matrix of regime labels (methods in rows)
            dates: Array of dates
            method_names: List of method names
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=regime_matrix,
            x=dates,
            y=method_names,
            colorscale=[
                [0, 'red'],
                [0.5, 'gray'],
                [1, 'green']
            ],
            zmin=0,
            zmax=2
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Method",
            height=600
        )
        
        return fig
    
    def plot_regime_transitions_3d(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        feature_names: List[str],
        title: str = "3D Regime Visualization"
    ) -> go.Figure:
        """Create 3D scatter plot of regime transitions.
        
        Args:
            features: Feature matrix (n_samples, 3)
            labels: Array of regime labels
            feature_names: List of feature names
            title: Plot title
        
        Returns:
            Plotly figure
        """
        if features.shape[1] != 3:
            raise ValueError("Features must have exactly 3 dimensions")
            
        fig = go.Figure()
        
        for regime in range(3):
            mask = labels == regime
            if not any(mask):
                continue
                
            fig.add_trace(go.Scatter3d(
                x=features[mask, 0],
                y=features[mask, 1],
                z=features[mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.colors[regime],
                    opacity=0.8
                ),
                name=f'Regime {regime}'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=feature_names[2]
            ),
            height=800
        )
        
        return fig
    
    def plot_regime_dashboard(
        self,
        dates: NDArray,
        prices: NDArray[np.float64],
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64],
        labels: Dict[str, NDArray[np.int64]],
        stats: Dict[str, Dict[int, Dict[str, float]]],
        title: str = "Market Regime Analysis Dashboard"
    ) -> go.Figure:
        """Create comprehensive dashboard for regime analysis.
        
        Args:
            dates: Array of dates
            prices: Array of prices
            returns: Array of returns
            volumes: Array of volumes
            labels: Dictionary of labels from different methods
            stats: Dictionary of statistics for each method
            title: Dashboard title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[
                "Price and Regimes", "Returns Distribution",
                "Volume Profile", "Regime Statistics",
                "Regime Transitions", "Method Comparison",
                "Regime Overlap", "Performance by Regime"
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Price and regimes
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                name='Price',
                line=dict(color='black', width=1)
            ),
            row=1, col=1
        )
        
        # Returns distribution
        for regime in range(3):
            mask = list(labels.values())[0] == regime
            if any(mask):
                fig.add_trace(
                    go.Histogram(
                        x=returns[mask],
                        name=f'Regime {regime}',
                        marker_color=self.colors[regime],
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Volume profile
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name='Volume',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Add other plots...
        # (Additional visualization code would go here)
        
        fig.update_layout(
            height=1200,
            title=title,
            showlegend=True
        )
        
        return fig
    
    def plot_regime_transition_matrix(
        self,
        labels: NDArray[np.int64],
        regime_names: Optional[List[str]] = None,
        title: str = "Regime Transition Matrix"
    ) -> go.Figure:
        """Plot regime transition probability matrix.
        
        Args:
            labels: Array of regime labels
            regime_names: Optional list of regime names
            title: Plot title
        
        Returns:
            Plotly figure
        """
        if regime_names is None:
            regime_names = ['Regime 0', 'Regime 1', 'Regime 2']
            
        # Calculate transition probabilities
        n_regimes = len(regime_names)
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(labels)-1):
            transitions[labels[i], labels[i+1]] += 1
            
        # Normalize by row
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions = transitions / row_sums
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=transitions,
            x=regime_names,
            y=regime_names,
            colorscale='RdYlGn',
            text=np.round(transitions, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            height=500,
            width=500
        )
        
        return fig
    
    def plot_regime_correlation_network(
        self,
        regime_labels: Dict[str, NDArray[np.int64]],
        title: str = "Regime Correlation Network"
    ) -> go.Figure:
        """Plot network graph of regime correlations.
        
        Args:
            regime_labels: Dictionary of regime labels from different methods
            title: Plot title
        
        Returns:
            Plotly figure
        """
        # Calculate correlations between regimes
        methods = list(regime_labels.keys())
        n_methods = len(methods)
        correlations = np.zeros((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                corr = np.corrcoef(
                    regime_labels[methods[i]],
                    regime_labels[methods[j]]
                )[0, 1]
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for method in methods:
            G.add_node(method)
        
        # Add edges with correlations
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                if abs(correlations[i, j]) > 0.3:  # Correlation threshold
                    G.add_edge(methods[i], methods[j], weight=abs(correlations[i, j]))
        
        # Get node positions
        pos = nx.spring_layout(G)
        
        # Create network plot
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=list(G.nodes()),
            textposition="bottom center"
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            height=600,
            width=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def plot_regime_performance_metrics(
        self,
        returns: NDArray[np.float64],
        labels: NDArray[np.int64],
        regime_names: Optional[List[str]] = None,
        rolling_window: int = 63,
        title: str = "Regime Performance Metrics"
    ) -> go.Figure:
        """Plot detailed performance metrics for each regime.
        
        Args:
            returns: Array of returns
            labels: Array of regime labels
            regime_names: Optional list of regime names
            rolling_window: Window for rolling metrics
            title: Plot title
        
        Returns:
            Plotly figure
        """
        if regime_names is None:
            regime_names = ['Regime 0', 'Regime 1', 'Regime 2']
            
        # Calculate rolling metrics
        rolling_ret = np.zeros_like(returns)
        rolling_vol = np.zeros_like(returns)
        rolling_sharpe = np.zeros_like(returns)
        
        for i in range(rolling_window, len(returns)):
            window_rets = returns[i-rolling_window:i]
            rolling_ret[i] = np.mean(window_rets) * 252  # Annualized
            rolling_vol[i] = np.std(window_rets) * np.sqrt(252)  # Annualized
            rolling_sharpe[i] = rolling_ret[i] / rolling_vol[i]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Rolling Returns by Regime",
                "Rolling Volatility by Regime",
                "Rolling Sharpe Ratio by Regime"
            ),
            vertical_spacing=0.1
        )
        
        # Plot metrics for each regime
        for regime in range(len(regime_names)):
            mask = labels == regime
            
            # Returns
            fig.add_trace(
                go.Scatter(
                    y=rolling_ret[mask],
                    name=f"{regime_names[regime]} Returns",
                    line=dict(color=self.colors[regime])
                ),
                row=1, col=1
            )
            
            # Volatility
            fig.add_trace(
                go.Scatter(
                    y=rolling_vol[mask],
                    name=f"{regime_names[regime]} Volatility",
                    line=dict(color=self.colors[regime])
                ),
                row=2, col=1
            )
            
            # Sharpe ratio
            fig.add_trace(
                go.Scatter(
                    y=rolling_sharpe[mask],
                    name=f"{regime_names[regime]} Sharpe",
                    line=dict(color=self.colors[regime])
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=900,
            title=title,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Annualized Return", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        return fig 