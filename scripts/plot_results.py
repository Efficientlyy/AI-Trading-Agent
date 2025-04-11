import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the results file relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Assumes scripts/ is one level down from root
RESULTS_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'backtest_portfolio_history.csv')

def plot_equity_curve(csv_path: str):
    """Loads backtest results and plots the equity curve."""
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return

    try:
        # Load the data, parsing 'timestamp' and setting it as the index
        history_df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

        if 'total_value' not in history_df.columns:
            print(f"Error: 'total_value' column not found in {csv_path}")
            return

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(history_df.index, history_df['total_value'], label='Portfolio Value')

        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

if __name__ == "__main__":
    print(f"Loading results from: {RESULTS_CSV_PATH}")
    plot_equity_curve(RESULTS_CSV_PATH)
