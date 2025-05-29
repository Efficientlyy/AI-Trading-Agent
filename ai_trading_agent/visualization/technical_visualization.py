"""
Technical Visualization Module

This module provides visualization tools for technical analysis components,
including chart patterns, indicators, and strategy signals.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from datetime import datetime
import os

# Setup logging
logger = logging.getLogger(__name__)

class TechnicalVisualizer:
    """
    Visualization tools for technical analysis components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the technical visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'theme': 'dark',
            'chart_height': 800,
            'indicator_height': 200,
            'show_volume': True,
            'show_patterns': True,
            'candlestick_colors': {
                'increasing': '#26A69A',
                'decreasing': '#EF5350'
            },
            'indicator_colors': {
                'sma': '#2962FF',
                'ema': '#FFEB3B',
                'bollinger': '#7B1FA2',
                'rsi': '#FF6D00',
                'macd': {
                    'line': '#2962FF',
                    'signal': '#FF6D00',
                    'histogram': '#26A69A'
                }
            }
        }
    
    def create_candlestick_chart(
        self, 
        df: pd.DataFrame, 
        title: str = 'Price Chart',
        show_volume: bool = None,
        indicators: List[Dict[str, Any]] = None,
        patterns: List[Dict[str, Any]] = None,
        signals: List[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Create a candlestick chart with optional indicators, patterns, and signals.
        
        Args:
            df: DataFrame with OHLCV data (must contain 'open', 'high', 'low', 'close', 'volume')
            title: Chart title
            show_volume: Whether to show volume (overrides config)
            indicators: List of indicators to display
            patterns: List of patterns to highlight
            signals: List of trading signals to mark
            
        Returns:
            Plotly figure object
        """
        # Set default values from config if not provided
        show_volume = show_volume if show_volume is not None else self.config['show_volume']
        
        # Determine the number of rows for subplots
        n_rows = 1  # Main price chart
        
        # Add rows for volume and indicators
        if show_volume:
            n_rows += 1
        
        # Count separate indicator panes (not overlaid)
        separate_indicators = 0
        if indicators:
            for ind in indicators:
                if ind.get('overlay', False) is False:
                    separate_indicators += 1
        
        n_rows += separate_indicators
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6] + [0.4 / (n_rows - 1)] * (n_rows - 1) if n_rows > 1 else [1],
            subplot_titles=[title] + ['Volume'] * (1 if show_volume else 0) + [''] * separate_indicators
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color=self.config['candlestick_colors']['increasing'],
                decreasing_line_color=self.config['candlestick_colors']['decreasing'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume chart if enabled
        current_row = 2
        if show_volume:
            colors = np.where(df['close'] >= df['open'], self.config['candlestick_colors']['increasing'], 
                              self.config['candlestick_colors']['decreasing'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    marker_color=colors,
                    name='Volume'
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Add indicators
        if indicators:
            for ind in indicators:
                ind_type = ind.get('type', '').lower()
                ind_data = ind.get('data', {})
                overlay = ind.get('overlay', False)
                
                if ind_type == 'sma' or ind_type == 'ema':
                    # Simple/Exponential Moving Average
                    period = ind.get('period', 20)
                    color = ind.get('color', self.config['indicator_colors'].get(ind_type, '#2962FF'))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('values', []),
                            mode='lines',
                            line=dict(color=color, width=1.5),
                            name=f"{ind_type.upper()}({period})"
                        ),
                        row=1 if overlay else current_row, col=1
                    )
                    
                    if not overlay:
                        current_row += 1
                
                elif ind_type == 'bollinger':
                    # Bollinger Bands
                    color = ind.get('color', self.config['indicator_colors'].get('bollinger', '#7B1FA2'))
                    alpha = ind.get('alpha', 0.3)
                    
                    # Add upper band
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('upper', []),
                            mode='lines',
                            line=dict(color=color, width=1),
                            name='Upper Band'
                        ),
                        row=1, col=1
                    )
                    
                    # Add middle band
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('middle', []),
                            mode='lines',
                            line=dict(color=color, width=1.5),
                            name='Middle Band'
                        ),
                        row=1, col=1
                    )
                    
                    # Add lower band
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('lower', []),
                            mode='lines',
                            line=dict(color=color, width=1),
                            name='Lower Band',
                            fill='tonexty',
                            fillcolor=f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha})'
                        ),
                        row=1, col=1
                    )
                
                elif ind_type == 'rsi':
                    # RSI
                    color = ind.get('color', self.config['indicator_colors'].get('rsi', '#FF6D00'))
                    period = ind.get('period', 14)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('values', []),
                            mode='lines',
                            line=dict(color=color, width=1.5),
                            name=f"RSI({period})"
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add overbought/oversold lines
                    overbought = ind.get('overbought', 70)
                    oversold = ind.get('oversold', 30)
                    
                    fig.add_shape(
                        type='line',
                        x0=df.index[0],
                        y0=overbought,
                        x1=df.index[-1],
                        y1=overbought,
                        line=dict(color='gray', width=1, dash='dash'),
                        row=current_row, col=1
                    )
                    
                    fig.add_shape(
                        type='line',
                        x0=df.index[0],
                        y0=oversold,
                        x1=df.index[-1],
                        y1=oversold,
                        line=dict(color='gray', width=1, dash='dash'),
                        row=current_row, col=1
                    )
                    
                    current_row += 1
                
                elif ind_type == 'macd':
                    # MACD
                    line_color = ind.get('line_color', self.config['indicator_colors']['macd']['line'])
                    signal_color = ind.get('signal_color', self.config['indicator_colors']['macd']['signal'])
                    hist_color = ind.get('hist_color', self.config['indicator_colors']['macd']['histogram'])
                    
                    # Add MACD line
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('macd', []),
                            mode='lines',
                            line=dict(color=line_color, width=1.5),
                            name='MACD'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add signal line
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ind_data.get('signal', []),
                            mode='lines',
                            line=dict(color=signal_color, width=1.5),
                            name='Signal'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add histogram
                    histogram = ind_data.get('histogram', [])
                    colors = np.where(np.array(histogram) >= 0, 
                                     self.config['candlestick_colors']['increasing'], 
                                     self.config['candlestick_colors']['decreasing'])
                    
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=histogram,
                            marker_color=colors,
                            name='Histogram'
                        ),
                        row=current_row, col=1
                    )
                    
                    current_row += 1
        
        # Add patterns
        if patterns and self.config['show_patterns']:
            for pattern in patterns:
                pattern_type = pattern.get('pattern', '')
                direction = pattern.get('direction', 'neutral')
                start_idx = pattern.get('start_idx', 0)
                end_idx = pattern.get('end_idx', 0)
                confidence = pattern.get('confidence', 0)
                
                # Get color based on direction
                color = 'green' if direction == 'bullish' else 'red' if direction == 'bearish' else 'gray'
                
                # Create pattern annotation
                fig.add_shape(
                    type='rect',
                    x0=df.index[start_idx],
                    y0=df['low'][start_idx:end_idx+1].min() * 0.99,  # 1% below min
                    x1=df.index[end_idx],
                    y1=df['high'][start_idx:end_idx+1].max() * 1.01,  # 1% above max
                    line=dict(color=color, width=1, dash='dot'),
                    fillcolor=f'rgba({0 if color != "red" else 255},{0 if color != "green" else 255},0,0.1)',
                    row=1, col=1
                )
                
                # Add pattern label
                fig.add_annotation(
                    x=df.index[start_idx + (end_idx - start_idx) // 2],
                    y=df['high'][start_idx:end_idx+1].max() * 1.03,
                    text=f"{pattern_type.replace('_', ' ').title()} ({int(confidence * 100)}%)",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor=color,
                    arrowsize=1,
                    arrowwidth=2,
                    row=1, col=1
                )
        
        # Add signals
        if signals:
            for signal in signals:
                signal_type = signal.get('type', '')
                position = signal.get('position', 0)
                direction = signal.get('direction', 'neutral')
                confidence = signal.get('confidence', 0)
                
                # Get color and symbol based on direction
                color = 'green' if direction == 'buy' else 'red' if direction == 'sell' else 'blue'
                symbol = 'triangle-up' if direction == 'buy' else 'triangle-down' if direction == 'sell' else 'circle'
                
                # Add signal marker
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[position]],
                        y=[df['low'][position] * 0.99 if direction == 'buy' else df['high'][position] * 1.01],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=10,
                            symbol=symbol,
                            line=dict(color='white', width=1)
                        ),
                        name=f"{signal_type.replace('_', ' ').title()} ({int(confidence * 100)}%)",
                        hoverinfo='text',
                        hovertext=f"{signal_type.replace('_', ' ').title()}: {direction.upper()} ({int(confidence * 100)}%)"
                    ),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=self.config['chart_height'],
            template='plotly_dark' if self.config['theme'] == 'dark' else 'plotly_white',
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, b=10, t=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Update Y-axis titles
        fig.update_yaxes(title_text='Price', row=1, col=1)
        
        if show_volume:
            fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """
        Save a chart to a file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format ('html', 'png', 'jpg', 'svg', 'pdf')
            
        Returns:
            Path to the saved file
        """
        if format == 'html':
            fig.write_html(filename)
        elif format in ['png', 'jpg', 'svg', 'pdf']:
            fig.write_image(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filename
    
    def create_multi_timeframe_chart(
        self,
        data_dict: Dict[str, pd.DataFrame],
        title: str = 'Multi-Timeframe Analysis',
        indicators: List[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Create a multi-timeframe chart for comparing price action across timeframes.
        
        Args:
            data_dict: Dictionary mapping timeframe names to DataFrames
            title: Chart title
            indicators: List of indicators to display (applied to each timeframe)
            
        Returns:
            Plotly figure object
        """
        n_timeframes = len(data_dict)
        
        fig = make_subplots(
            rows=n_timeframes,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=[f"{tf.upper()} Timeframe" for tf in data_dict.keys()]
        )
        
        row = 1
        for timeframe, df in data_dict.items():
            # Add candlestick chart for this timeframe
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    increasing_line_color=self.config['candlestick_colors']['increasing'],
                    decreasing_line_color=self.config['candlestick_colors']['decreasing'],
                    name=f'{timeframe} Price'
                ),
                row=row, col=1
            )
            
            # Add indicators if provided
            if indicators:
                for ind in indicators:
                    ind_type = ind.get('type', '').lower()
                    if ind_type == 'sma':
                        period = ind.get('period', 20)
                        color = ind.get('color', self.config['indicator_colors'].get('sma', '#2962FF'))
                        
                        # Calculate SMA for this timeframe
                        sma = df['close'].rolling(window=period).mean()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=sma,
                                mode='lines',
                                line=dict(color=color, width=1.5),
                                name=f"{timeframe} SMA({period})"
                            ),
                            row=row, col=1
                        )
            
            row += 1
        
        # Update layout
        fig.update_layout(
            height=300 * n_timeframes,
            title=title,
            template='plotly_dark' if self.config['theme'] == 'dark' else 'plotly_white',
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, b=10, t=40 + 20 * n_timeframes),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def create_pattern_summary(
        self,
        patterns: List[Dict[str, Any]],
        df: pd.DataFrame = None
    ) -> go.Figure:
        """
        Create a visual summary of detected patterns.
        
        Args:
            patterns: List of pattern dictionaries
            df: Optional DataFrame with price data for context
            
        Returns:
            Plotly figure object
        """
        if not patterns:
            return go.Figure()
        
        # Group patterns by type
        pattern_types = {}
        for p in patterns:
            p_type = p.get('pattern', 'unknown')
            if p_type not in pattern_types:
                pattern_types[p_type] = []
            pattern_types[p_type].append(p)
        
        # Create figure with bar chart of pattern counts and confidence
        pattern_names = list(pattern_types.keys())
        pattern_counts = [len(pattern_types[p]) for p in pattern_names]
        pattern_confidence = [sum(p['confidence'] for p in pattern_types[p]) / len(pattern_types[p]) for p in pattern_names]
        
        fig = go.Figure()
        
        # Add count bars
        fig.add_trace(go.Bar(
            x=pattern_names,
            y=pattern_counts,
            name='Count',
            marker_color='rgba(58, 71, 80, 0.6)',
            text=pattern_counts,
            textposition='auto'
        ))
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=pattern_names,
            y=pattern_confidence,
            mode='lines+markers',
            name='Avg Confidence',
            yaxis='y2',
            line=dict(color='rgb(82, 188, 163)', width=2),
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title='Pattern Detection Summary',
            template='plotly_dark' if self.config['theme'] == 'dark' else 'plotly_white',
            xaxis=dict(
                title='Pattern Type',
                tickangle=-45
            ),
            yaxis=dict(
                title='Count',
                titlefont=dict(color='rgb(58, 71, 80)'),
                tickfont=dict(color='rgb(58, 71, 80)')
            ),
            yaxis2=dict(
                title='Average Confidence',
                titlefont=dict(color='rgb(82, 188, 163)'),
                tickfont=dict(color='rgb(82, 188, 163)'),
                anchor='x',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            barmode='group',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def export_chart_data(
        self, 
        df: pd.DataFrame, 
        indicators: List[Dict[str, Any]] = None,
        patterns: List[Dict[str, Any]] = None,
        filename: str = None
    ) -> Dict[str, Any]:
        """
        Export chart data as JSON for external rendering.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators
            patterns: List of patterns
            filename: Optional filename to save JSON
            
        Returns:
            Dictionary with chart data
        """
        # Convert DataFrame to dictionary
        df_dict = {
            'dates': df.index.astype(str).tolist(),
            'ohlcv': {
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'close': df['close'].tolist(),
                'volume': df['volume'].tolist() if 'volume' in df.columns else []
            }
        }
        
        # Add indicators
        if indicators:
            df_dict['indicators'] = indicators
        
        # Add patterns
        if patterns:
            df_dict['patterns'] = patterns
        
        # Save to file if filename provided
        if filename:
            with open(filename, 'w') as f:
                json.dump(df_dict, f)
        
        return df_dict
