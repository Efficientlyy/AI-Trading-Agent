"""Market Regime Detection Web UI.

This module provides a web interface for the Market Regime Detection API.
"""

import os
import json
import requests
import io
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from urllib.parse import urljoin
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file, Response
import pdfkit

# Configuration
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')
DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
DEFAULT_PERIOD = '3y'
DEFAULT_LOOKBACK = 60
DEFAULT_METHODS = ['volatility', 'momentum', 'hmm']

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'development_secret_key')

# Session lifetime
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Helper functions
def get_api_url(endpoint):
    """Build API URL from endpoint."""
    return urljoin(API_BASE_URL, endpoint)

def format_date(date_str):
    """Format date string for display."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%b %d, %Y')
    except ValueError:
        return date_str

# Routes
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', 
                          default_symbols=DEFAULT_SYMBOLS,
                          default_period=DEFAULT_PERIOD,
                          default_lookback=DEFAULT_LOOKBACK,
                          default_methods=DEFAULT_METHODS)

@app.route('/detect_regimes', methods=['POST'])
def detect_regimes():
    """Call the API to detect regimes."""
    try:
        # Get form data
        symbol = request.form.get('symbol', DEFAULT_SYMBOLS[0])
        period = request.form.get('period', DEFAULT_PERIOD)
        lookback = int(request.form.get('lookback', DEFAULT_LOOKBACK))
        methods = request.form.getlist('methods') or DEFAULT_METHODS
        include_stats = 'include_stats' in request.form
        include_viz = 'include_viz' in request.form
        
        # Prepare API request
        api_url = get_api_url('api/v1/detect')
        payload = {
            'symbol': symbol,
            'period': period,
            'lookback_window': lookback,
            'methods': methods,
            'include_statistics': include_stats,
            'include_visualizations': include_viz
        }
        
        # Call API
        response = requests.post(api_url, json=payload)
        if response.status_code != 200:
            flash(f"API Error: {response.status_code} - {response.text}", "error")
            return redirect(url_for('index'))
        
        # Store results in session
        result = response.json()
        session['detection_result'] = result
        
        # Redirect to results page
        return redirect(url_for('results'))
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display regime detection results."""
    result = session.get('detection_result')
    if not result:
        flash("No detection results available. Please submit a new request.", "warning")
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                          result=result,
                          format_date=format_date)

@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    """Backtest a regime-based trading strategy."""
    if request.method == 'GET':
        # Show backtest form
        return render_template('backtest.html',
                              default_symbols=DEFAULT_SYMBOLS,
                              default_period=DEFAULT_PERIOD,
                              default_methods=DEFAULT_METHODS)
    
    try:
        # Get basic form data
        symbol = request.form.get('symbol', DEFAULT_SYMBOLS[0])
        period = request.form.get('period', DEFAULT_PERIOD)
        strategy = request.form.get('strategy', 'trend_following')
        regime_methods = request.form.getlist('methods') or DEFAULT_METHODS
        initial_capital = float(request.form.get('initial_capital', 10000))
        
        # Get advanced options
        position_sizing = request.form.get('position_sizing', 'fixed')
        max_position_size = float(request.form.get('max_position_size', 100)) / 100.0  # Convert to decimal
        
        # Get optional stop loss and take profit values
        stop_loss_pct = None
        if request.form.get('stop_loss_pct'):
            stop_loss_pct = float(request.form.get('stop_loss_pct')) / 100.0  # Convert to decimal
            
        take_profit_pct = None
        if request.form.get('take_profit_pct'):
            take_profit_pct = float(request.form.get('take_profit_pct')) / 100.0  # Convert to decimal
            
        # Check if transaction costs should be included
        include_transaction_costs = 'include_transaction_costs' in request.form
        
        # Download market data
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                flash(f"No data found for symbol {symbol} with period {period}", "error")
                return redirect(url_for('backtest'))
                
        except Exception as e:
            flash(f"Error downloading data: {str(e)}", "error")
            return redirect(url_for('backtest'))
        
        # Prepare data points
        data_points = []
        for index, row in df.iterrows():
            data_point = {
                "date": index.strftime('%Y-%m-%dT%H:%M:%S'),
                "price": float(row["Close"]),
                "volume": float(row["Volume"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "return_value": float(row["Close"] / df["Close"].shift(1) - 1) if not pd.isna(row["Close"] / df["Close"].shift(1) - 1) else 0
            }
            data_points.append(data_point)
        
        # Create market data structure
        market_data = {
            "symbol": symbol,
            "data": data_points
        }
        
        # Prepare API request
        api_url = get_api_url('api/v1/backtest')
        payload = {
            'market_data': market_data,
            'strategy_type': strategy,
            'regime_methods': regime_methods,
            'initial_capital': initial_capital,
            'position_sizing': position_sizing,
            'max_position_size': max_position_size,
        }
        
        # Add optional parameters only if they have values
        if stop_loss_pct is not None:
            payload['stop_loss_pct'] = stop_loss_pct
            
        if take_profit_pct is not None:
            payload['take_profit_pct'] = take_profit_pct
            
        payload['include_transaction_costs'] = include_transaction_costs
        
        # Call API
        response = requests.post(api_url, json=payload)
        if response.status_code != 200:
            flash(f"API Error: {response.status_code} - {response.text}", "error")
            return redirect(url_for('backtest'))
        
        # Store results in session
        result = response.json()
        session['backtest_result'] = result
        
        # Redirect to backtest results page
        return redirect(url_for('backtest_results'))
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('backtest'))

@app.route('/backtest_results')
def backtest_results():
    """Display backtest results."""
    result = session.get('backtest_result')
    if not result:
        flash("No backtest results available. Please submit a new request.", "warning")
        return redirect(url_for('backtest'))
    
    return render_template('backtest_results.html', 
                          result=result,
                          format_date=format_date)

@app.route('/generate_backtest_report')
def generate_backtest_report():
    """Generate a PDF report for the backtest results."""
    result = session.get('backtest_result')
    if not result:
        flash("No backtest results available. Please run a backtest first.", "warning")
        return redirect(url_for('backtest'))
    
    try:
        # Call the API to generate a PDF report
        api_url = get_api_url('api/v1/generate_report')
        response = requests.post(api_url, json=result)
        
        if response.status_code != 200:
            flash(f"Error generating report: {response.text}", "error")
            return redirect(url_for('backtest_results'))
        
        # Get the PDF from the response
        pdf_content = response.content
        
        # Prepare the PDF to send back to the client
        buffer = io.BytesIO(pdf_content)
        buffer.seek(0)
        
        # Return the PDF file
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"backtest_report_{result['symbol']}.pdf"
        )
        
    except Exception as e:
        flash(f"Error generating report: {str(e)}", "error")
        return redirect(url_for('backtest_results'))

@app.route('/api_docs')
def api_docs():
    """API documentation page."""
    return render_template('api_docs.html')

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    """Generate a PDF report of backtest results."""
    try:
        # Get data from request
        data = request.json
        
        # Create HTML for PDF
        html = render_template(
            'pdf_report.html',
            symbol=data.get('symbol', ''),
            strategy=data.get('strategy', ''),
            metrics=data.get('metrics', {}),
            chart_urls=data.get('chartUrls', []),
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Configure PDF options
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': 'UTF-8',
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        # Generate PDF
        pdf = pdfkit.from_string(html, False, options=options)
        
        # Create response
        pdf_io = io.BytesIO(pdf)
        pdf_io.seek(0)
        
        # Return PDF file
        return send_file(
            pdf_io,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{data.get('symbol', 'backtest')}_{data.get('strategy', 'strategy')}_report.pdf"
        )
    except Exception as e:
        app.logger.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    """404 error handler."""
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler."""
    return render_template('errors/500.html'), 500

# Run the application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug) 