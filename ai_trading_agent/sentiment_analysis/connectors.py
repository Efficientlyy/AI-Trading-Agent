# ai_trading_agent/sentiment_analysis/connectors.py
import pandas as pd
import os

def load_sentiment_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads sentiment data from a CSV file into a pandas DataFrame.

    The CSV file is expected to have columns: 'timestamp', 'text', 'symbol'.
    'timestamp' should be parsable by pandas.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the sentiment data.
        Returns an empty DataFrame if the file is not found or empty.

    Raises:
        ValueError: If the required columns are missing.
        Exception: For other pandas read_csv errors.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['timestamp', 'text', 'symbol'])

    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])

        # Check for required columns
        required_columns = ['timestamp', 'text', 'symbol']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file missing required columns. Expected: {required_columns}, Found: {list(df.columns)}")

        if df.empty:
            print(f"Warning: CSV file at {file_path} is empty.")
            # Ensure correct columns even if empty
            return pd.DataFrame(columns=required_columns)

        # Ensure timestamp is timezone-naive or convert to UTC for consistency
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None) # Convert to timezone-naive

        # Optional: Sort by timestamp if needed
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        return df

    except ValueError as ve:
        print(f"Error loading CSV: {ve}")
        raise # Re-raise the ValueError
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        raise # Re-raise other exceptions
