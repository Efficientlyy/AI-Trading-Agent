
# Mock data loader for testing
def load_mock_data(filename):
    # Return empty data for testing
    if "sentiment" in filename:
        return {"positive": 0.5, "negative": 0.2, "neutral": 0.3}
    elif "market" in filename:
        return {"price": 50000, "volume": 1000000}
    else:
        return {}
