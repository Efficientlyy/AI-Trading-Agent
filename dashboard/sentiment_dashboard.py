"""
Sentiment Analysis Dashboard

This module provides a Streamlit dashboard for visualizing sentiment data
and the resulting trading signals.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from ai_trading_agent.data_sources.alpha_vantage_client import AlphaVantageClient
from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Crypto Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None
if 'signals' not in st.session_state:
    st.session_state.signals = None

def get_sentiment_data(assets, days_back):
    """Fetch sentiment data for the specified assets."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        st.error("No Alpha Vantage API key found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        return None
    
    client = AlphaVantageClient(api_key=api_key)
    analyzer = SentimentAnalyzer(client)
    
    all_data = []
    
    for asset in assets:
        with st.spinner(f"Fetching sentiment data for {asset}..."):
            try:
                # Get sentiment data
                sentiment_data = analyzer.get_sentiment_for_asset(asset, days_back=days_back)
                
                if sentiment_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(sentiment_data)
                    df['asset'] = asset
                    all_data.append(df)
            except Exception as e:
                st.error(f"Error fetching data for {asset}: {e}")
    
    if not all_data:
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert time_published to datetime
    combined_df['time_published'] = pd.to_datetime(combined_df['time_published'])
    
    # Sort by time
    combined_df = combined_df.sort_values('time_published')
    
    return combined_df

def generate_signals(sentiment_df, window_size=3, threshold=0.2):
    """Generate trading signals from sentiment data."""
    if sentiment_df is None or len(sentiment_df) == 0:
        return None
    
    signals = []
    
    # Group by asset
    for asset, group in sentiment_df.groupby('asset'):
        # Sort by time
        group = group.sort_values('time_published')
        
        # Calculate rolling average sentiment
        group['rolling_sentiment'] = group['overall_sentiment_score'].rolling(window=window_size).mean()
        
        # Generate signals
        group['signal'] = 0
        group.loc[group['rolling_sentiment'] > threshold, 'signal'] = 1
        group.loc[group['rolling_sentiment'] < -threshold, 'signal'] = -1
        
        signals.append(group)
    
    # Combine all signals
    signals_df = pd.concat(signals, ignore_index=True)
    
    return signals_df

def plot_sentiment_trends(df):
    """Plot sentiment trends over time."""
    if df is None or len(df) == 0:
        st.warning("No sentiment data available to plot.")
        return
    
    # Group by asset and date
    df['date'] = df['time_published'].dt.date
    daily_sentiment = df.groupby(['asset', 'date'])['overall_sentiment_score'].mean().reset_index()
    
    # Plot
    fig = px.line(
        daily_sentiment, 
        x='date', 
        y='overall_sentiment_score', 
        color='asset',
        title='Daily Average Sentiment Score by Asset',
        labels={'overall_sentiment_score': 'Sentiment Score', 'date': 'Date'},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_distribution(df):
    """Plot sentiment distribution."""
    if df is None or len(df) == 0:
        st.warning("No sentiment data available to plot.")
        return
    
    fig = px.box(
        df, 
        x='asset', 
        y='overall_sentiment_score',
        title='Sentiment Score Distribution by Asset',
        labels={'overall_sentiment_score': 'Sentiment Score', 'asset': 'Asset'},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_signals(df):
    """Plot trading signals based on sentiment."""
    if df is None or len(df) == 0:
        st.warning("No signal data available to plot.")
        return
    
    # Group by asset and date
    df['date'] = df['time_published'].dt.date
    daily_signals = df.groupby(['asset', 'date']).agg({
        'rolling_sentiment': 'mean',
        'signal': 'last'
    }).reset_index()
    
    # Create subplots
    fig = go.Figure()
    
    for asset, group in daily_signals.groupby('asset'):
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=group['date'],
            y=group['rolling_sentiment'],
            mode='lines',
            name=f'{asset} Sentiment',
            line=dict(width=2)
        ))
        
        # Add buy signals
        buy_signals = group[group['signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['rolling_sentiment'],
            mode='markers',
            name=f'{asset} Buy Signal',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ))
        
        # Add sell signals
        sell_signals = group[group['signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals['date'],
            y=sell_signals['rolling_sentiment'],
            mode='markers',
            name=f'{asset} Sell Signal',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ))
    
    fig.update_layout(
        title='Sentiment-Based Trading Signals',
        xaxis_title='Date',
        yaxis_title='Rolling Sentiment Score',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_news_feed(df, limit=10):
    """Display recent news articles with sentiment scores."""
    if df is None or len(df) == 0:
        st.warning("No news data available to display.")
        return
    
    # Sort by time (most recent first)
    recent_news = df.sort_values('time_published', ascending=False).head(limit)
    
    st.subheader(f"Recent News Articles (Last {limit})")
    
    for _, article in recent_news.iterrows():
        # Create a card-like container for each article
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Article title and link
                st.markdown(f"### [{article['title']}]({article['url']})")
                
                # Publication time and source
                st.markdown(f"**Published:** {article['time_published']} | **Source:** {article['source']}")
                
                # Summary
                if 'summary' in article and article['summary']:
                    st.markdown(f"**Summary:** {article['summary']}")
            
            with col2:
                # Sentiment score
                sentiment_score = float(article['overall_sentiment_score'])
                sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "gray"
                
                st.markdown(f"""
                <div style="background-color: {sentiment_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                    <h2>{sentiment_score:.2f}</h2>
                    <p>{article['overall_sentiment_label']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Horizontal line
            st.markdown("---")

def main():
    """Main dashboard function."""
    st.title("Crypto Sentiment Analysis Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Asset selection
    available_assets = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "DOGE", "LINK"]
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        available_assets,
        default=["BTC", "ETH"]
    )
    
    # Time range selection
    days_back = st.sidebar.slider(
        "Days to Look Back",
        min_value=1,
        max_value=30,
        value=7
    )
    
    # Signal parameters
    st.sidebar.subheader("Signal Parameters")
    window_size = st.sidebar.slider(
        "Rolling Window Size",
        min_value=1,
        max_value=10,
        value=3
    )
    
    threshold = st.sidebar.slider(
        "Signal Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05
    )
    
    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        if not selected_assets:
            st.warning("Please select at least one asset.")
        else:
            with st.spinner("Fetching sentiment data..."):
                sentiment_data = get_sentiment_data(selected_assets, days_back)
                st.session_state.sentiment_data = sentiment_data
                
                if sentiment_data is not None:
                    st.success(f"Successfully fetched data for {', '.join(selected_assets)}")
                    
                    # Generate signals
                    signals = generate_signals(sentiment_data, window_size, threshold)
                    st.session_state.signals = signals
    
    # Main content
    if st.session_state.sentiment_data is not None:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Trends", "Sentiment Distribution", "Trading Signals", "News Feed"])
        
        with tab1:
            plot_sentiment_trends(st.session_state.sentiment_data)
        
        with tab2:
            plot_sentiment_distribution(st.session_state.sentiment_data)
        
        with tab3:
            plot_signals(st.session_state.signals)
        
        with tab4:
            display_news_feed(st.session_state.sentiment_data)
    else:
        st.info("Select assets and click 'Fetch Data' to load sentiment analysis.")
        
        # Sample image or placeholder
        st.image("https://www.alphavantage.co/static/img/banner.png", caption="Powered by Alpha Vantage API")

if __name__ == "__main__":
    main()
