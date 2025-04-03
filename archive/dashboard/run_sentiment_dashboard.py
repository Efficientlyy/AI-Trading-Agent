"""
Sentiment Dashboard Runner

This script runs the sentiment analysis dashboard with proper path configuration
and mock data generation when necessary.
"""

import os
import sys
import logging
import random
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add the project root to Python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mock_sentiment_data():
    """Generate mock sentiment data for demonstration purposes."""
    # Create dashboard_templates directory if it doesn't exist
    templates_dir = Path("dashboard_templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Create a mock sentiment dashboard HTML template if it doesn't exist
    sentiment_template_path = templates_dir / "sentiment_dashboard.html"
    if not sentiment_template_path.exists():
        logger.info("Creating mock sentiment dashboard template...")
        with open(sentiment_template_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f8fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #1a2c42;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        h1 {
            margin: 0;
            font-size: 2em;
        }
        h2 {
            margin-top: 0;
            color: #1a2c42;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }
        .sentiment-score {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .positive { color: #28a745; }
        .neutral { color: #6c757d; }
        .negative { color: #dc3545; }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .metric-row:last-child {
            border-bottom: none;
        }
        .metric-name {
            font-weight: bold;
        }
        .metric-value {
            font-weight: bold;
        }
        .refresh-message {
            text-align: center;
            color: #6c757d;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #f0f0f0;
        }
        th {
            background-color: #f5f8fa;
        }
    </style>
</head>
<body>
    <header>
        <h1>Sentiment Analysis Dashboard</h1>
    </header>
    <div class="container">
        <div class="dashboard">
            <!-- Overall Sentiment Card -->
            <div class="card">
                <h2>Overall Market Sentiment</h2>
                <div class="sentiment-score positive">{{ overall_score }}</div>
                <div>
                    <div class="metric-row">
                        <span class="metric-name">Social Media:</span>
                        <span class="metric-value">{{ social_score }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">News:</span>
                        <span class="metric-value">{{ news_score }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">On-chain:</span>
                        <span class="metric-value">{{ onchain_score }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Market:</span>
                        <span class="metric-value">{{ market_score }}</span>
                    </div>
                </div>
            </div>

            <!-- Sentiment Trends Card -->
            <div class="card">
                <h2>Sentiment Trends (7-Day)</h2>
                <p>Sentiment trend visualization would appear here.</p>
                <div>
                    <div class="metric-row">
                        <span class="metric-name">Trend Direction:</span>
                        <span class="metric-value">{{ trend_direction }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Volatility:</span>
                        <span class="metric-value">{{ sentiment_volatility }}</span>
                    </div>
                </div>
            </div>

            <!-- Price Correlation Card -->
            <div class="card">
                <h2>Price-Sentiment Correlation</h2>
                <p>Price-sentiment correlation visualization would appear here.</p>
                <div>
                    <div class="metric-row">
                        <span class="metric-name">7-Day Correlation:</span>
                        <span class="metric-value">{{ seven_day_correlation }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">30-Day Correlation:</span>
                        <span class="metric-value">{{ thirty_day_correlation }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Lead-Lag Relationship:</span>
                        <span class="metric-value">{{ lead_lag }}</span>
                    </div>
                </div>
            </div>

            <!-- News Sentiment Card -->
            <div class="card">
                <h2>News Sentiment Analysis</h2>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Headline</th>
                        <th>Sentiment</th>
                    </tr>
                    {% for article in news_articles %}
                    <tr>
                        <td>{{ article.source }}</td>
                        <td>{{ article.headline }}</td>
                        <td>{{ article.sentiment }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <!-- Social Media Sentiment Card -->
            <div class="card">
                <h2>Social Media Sentiment</h2>
                <div>
                    <div class="metric-row">
                        <span class="metric-name">Twitter:</span>
                        <span class="metric-value">{{ twitter_sentiment }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Reddit:</span>
                        <span class="metric-value">{{ reddit_sentiment }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Trading Forums:</span>
                        <span class="metric-value">{{ forum_sentiment }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Trending Topics:</span>
                        <span class="metric-value">{{ trending_topics }}</span>
                    </div>
                </div>
            </div>

            <!-- Sentiment Signals Card -->
            <div class="card">
                <h2>Trading Signals Based on Sentiment</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Signal</th>
                        <th>Strength</th>
                    </tr>
                    {% for signal in trading_signals %}
                    <tr>
                        <td>{{ signal.asset }}</td>
                        <td>{{ signal.signal }}</td>
                        <td>{{ signal.strength }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        <div class="refresh-message">
            <p>Last updated: {{ current_time }} | Refresh the page to update data</p>
        </div>
    </div>
</body>
</html>""")

    # Create mock sentiment data
    logger.info("Generating mock sentiment data...")
    assets = ["BTC", "ETH", "SOL", "ADA", "DOT"]
    
    # Mock sentiment scores
    overall_score = round(random.uniform(60, 80), 1)
    social_score = round(random.uniform(50, 90), 1)
    news_score = round(random.uniform(40, 85), 1)
    onchain_score = round(random.uniform(45, 95), 1)
    market_score = round(random.uniform(30, 90), 1)
    
    # Mock sentiment trends
    trend_directions = ["Improving", "Declining", "Stable", "Volatile"]
    trend_direction = random.choice(trend_directions)
    sentiment_volatility = f"{round(random.uniform(0.3, 2.5), 2)}"
    
    # Mock correlations
    seven_day_correlation = round(random.uniform(-0.8, 0.8), 2)
    thirty_day_correlation = round(random.uniform(-0.7, 0.9), 2)
    lead_lag_options = ["Sentiment leads price", "Price leads sentiment", "No clear lead-lag"]
    lead_lag = random.choice(lead_lag_options)
    
    # Mock news articles
    news_sources = ["CryptoNews", "CoinDesk", "Bloomberg", "Reuters", "WSJ"]
    headlines = [
        "Bitcoin Reaches New Monthly High As Institutional Interest Grows",
        "Regulatory Framework for Cryptocurrencies Being Developed",
        "Major Bank Announces Cryptocurrency Trading Platform",
        "Ethereum Update Postponed Due to Security Concerns",
        "Central Banks Consider Digital Currency Development"
    ]
    sentiment_options = ["Positive", "Negative", "Neutral", "Very Positive", "Very Negative"]
    
    news_articles = []
    for i in range(5):
        news_articles.append({
            "source": random.choice(news_sources),
            "headline": random.choice(headlines),
            "sentiment": random.choice(sentiment_options)
        })
    
    # Mock social media sentiment
    twitter_sentiment = f"{round(random.uniform(30, 90), 1)}"
    reddit_sentiment = f"{round(random.uniform(30, 90), 1)}"
    forum_sentiment = f"{round(random.uniform(30, 90), 1)}"
    trending_topics = "DeFi, NFTs, Bitcoin ETF"
    
    # Mock trading signals
    signal_options = ["Buy", "Sell", "Hold", "Strong Buy", "Strong Sell"]
    strength_options = ["Low", "Medium", "High", "Very High"]
    
    trading_signals = []
    for asset in assets:
        trading_signals.append({
            "asset": asset,
            "signal": random.choice(signal_options),
            "strength": random.choice(strength_options)
        })
    
    # Create the data dictionary
    mock_data = {
        "overall_score": f"{overall_score}",
        "social_score": f"{social_score}",
        "news_score": f"{news_score}",
        "onchain_score": f"{onchain_score}",
        "market_score": f"{market_score}",
        "trend_direction": trend_direction,
        "sentiment_volatility": sentiment_volatility,
        "seven_day_correlation": f"{seven_day_correlation}",
        "thirty_day_correlation": f"{thirty_day_correlation}",
        "lead_lag": lead_lag,
        "news_articles": news_articles,
        "twitter_sentiment": twitter_sentiment,
        "reddit_sentiment": reddit_sentiment,
        "forum_sentiment": forum_sentiment,
        "trending_topics": trending_topics,
        "trading_signals": trading_signals,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return mock_data

def main():
    """Run the sentiment dashboard with FastAPI."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the sentiment analysis dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server to")
    args = parser.parse_args()
    
    # Import required packages
    try:
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
    except ImportError as e:
        logger.error(f"Error importing required packages: {e}")
        logger.error("Please make sure you have installed all required packages:")
        logger.error("pip install fastapi uvicorn jinja2")
        sys.exit(1)
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("dashboard_templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Set up templates
    templates = Jinja2Templates(directory=str(templates_dir))
    
    # Create a FastAPI app
    app = FastAPI(title="Sentiment Analysis Dashboard")
    
    # Generate mock data
    mock_data = generate_mock_sentiment_data()
    
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Redirect to sentiment dashboard."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/sentiment" />
        </head>
        <body>
            <p>Redirecting to <a href="/sentiment">Sentiment Dashboard</a>...</p>
        </body>
        </html>
        """
    
    @app.get("/sentiment", response_class=HTMLResponse)
    async def get_sentiment_dashboard(request: Request):
        """Render the sentiment dashboard."""
        # Generate fresh mock data
        mock_data["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Use realistic market patterns based on user memory
        # This implements the "Realistic Market Patterns" and "Market Regime Simulation" features
        current_pattern = random.choice(["bull", "bear", "sideways", "volatile", "recovery", "crash"])
        
        if current_pattern == "bull":
            mock_data["overall_score"] = f"{round(random.uniform(70, 90), 1)}"
            mock_data["trend_direction"] = "Improving"
            mock_data["seven_day_correlation"] = f"{round(random.uniform(0.5, 0.9), 2)}"
        elif current_pattern == "bear":
            mock_data["overall_score"] = f"{round(random.uniform(20, 45), 1)}"
            mock_data["trend_direction"] = "Declining"
            mock_data["seven_day_correlation"] = f"{round(random.uniform(0.4, 0.8), 2)}"
        elif current_pattern == "volatile":
            mock_data["overall_score"] = f"{round(random.uniform(40, 70), 1)}"
            mock_data["trend_direction"] = "Volatile"
            mock_data["sentiment_volatility"] = f"{round(random.uniform(1.5, 3.0), 2)}"
        
        # Apply "Correlated Metrics" based on user memory
        # Ensure metrics are realistically correlated with market conditions
        overall_score_value = float(mock_data["overall_score"])
        if overall_score_value > 70:  # Positive sentiment
            # Higher likelihood of buy signals in bullish sentiment
            for signal in mock_data["trading_signals"]:
                if random.random() < 0.7:  # 70% chance
                    signal["signal"] = random.choice(["Buy", "Strong Buy"])
                    signal["strength"] = random.choice(["Medium", "High", "Very High"])
        elif overall_score_value < 40:  # Negative sentiment
            # Higher likelihood of sell signals in bearish sentiment
            for signal in mock_data["trading_signals"]:
                if random.random() < 0.7:  # 70% chance
                    signal["signal"] = random.choice(["Sell", "Strong Sell"])
                    signal["strength"] = random.choice(["Medium", "High", "Very High"])
        
        return templates.TemplateResponse("sentiment_dashboard.html", {"request": request, **mock_data})
    
    @app.get("/sentiment/api", response_class=JSONResponse)
    async def get_sentiment_data():
        """Get sentiment dashboard data as JSON for API clients."""
        # Generate fresh mock data
        mock_data["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return mock_data
    
    # Start the server
    logger.info(f"Starting sentiment dashboard at http://{args.host}:{args.port}/sentiment")
    logger.info("Press Ctrl+C to stop the dashboard")
    
    # Run the app with uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
