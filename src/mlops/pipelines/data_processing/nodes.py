import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(raw_filepath: str) -> pd.DataFrame:
    """Loads data from 'yahoo_stock.csv'."""
    df = pd.read_csv(raw_filepath)
    df["Date"] = pd.to_datetime(
        df["Date"]
    )  # Convert the 'Date' column to datetime format
    df = df.sort_values(by="Date")  # Sort data by date
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates moving averages and percentage returns."""
    df["MA7"] = df["Close"].rolling(window=7).mean()  # 7-day moving average
    df["MA30"] = df["Close"].rolling(window=30).mean()  # 30-day moving average
    df["Return"] = df["Close"].pct_change()  # Percentage change in closing price

    df["feature1"] = df["Return"].rolling(window=7).mean()  # 7-day rolling mean of returns
    df["feature2"] = df["MA7"] - df["MA30"]  # Difference between 7-day and 30-day moving averages
    
    # Remove missing values resulting from moving averages and return calculations
    df = df.dropna().reset_index(drop=True)
    return df


def create_sequences_and_scale(df: pd.DataFrame, features: list, seq_length: int):
    """
    Normalizes the data and creates time series sequences for the LSTM model.
    Returns X (features), y (target), and the scaler.
    """
    # Check if features is a list and ensure it's not empty
    if isinstance(features, pd.DataFrame):
        raise ValueError("The 'features' argument should be a list, not a DataFrame.")
    
    if not features:
        raise ValueError("Features list is empty. Please provide a valid list of features.")
    
    # Check if 'Close' is in the DataFrame
    if 'Close' not in df.columns:
        raise ValueError("'Close' column is missing in the DataFrame.")

    # Ensure that all columns in features exist in the DataFrame
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"The following features are missing in the DataFrame: {missing_features}")

    # Normalize the selected features and the target ('Close' price)
    scaler = MinMaxScaler()
    print(f"Normalizing columns: {features + ['Close']}")  # Debugging line
    data = scaler.fit_transform(df[features + ["Close"]])

    # Create sequences from the normalized data
    X, y = create_sequences(data, seq_length)

    return X, y, scaler


def create_sequences(data: np.ndarray, seq_length: int):
    """
    Creates sequences of length `seq_length` for features and target.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length, :-1])  # All columns except 'Close' (features)
        y.append(data[i + seq_length, -1])  # The 'Close' price column (target)
    return np.array(X), np.array(y)


def split_data(X, y, test_ratio: float):
    """
    Splits the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    return X_train, X_test, y_train, y_test
