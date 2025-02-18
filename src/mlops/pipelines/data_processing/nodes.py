import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(raw_filepath: str) -> pd.DataFrame:
    """Charge les données depuis 'yahoo_stock.csv'."""
    df = pd.read_csv(raw_filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les moyennes mobiles et la variation en pourcentage."""
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['Return'] = df['Close'].pct_change()
    # On supprime les valeurs manquantes dues aux moyennes mobiles et au calcul de Return
    df = df.dropna().reset_index(drop=True)
    return df

def create_sequences_and_scale(df: pd.DataFrame, features: list, seq_length: int):
    """
    Normalise les données et crée des séquences temporelles pour le modèle LSTM.
    Retourne X et y.
    """
    scaler = MinMaxScaler()
    # On normalise les features et la target ('Close')
    data = scaler.fit_transform(df[features + ['Close']])
    X, y = create_sequences(data, seq_length)
    return X, y, scaler

def create_sequences(data: np.ndarray, seq_length: int):
    """
    Crée des séquences de taille `seq_length` pour les features et la target.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # toutes les colonnes sauf 'Close'
        y.append(data[i+seq_length, -1])       # la colonne 'Close'
    return np.array(X), np.array(y)


def split_data(X, y, test_ratio: float):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    return X_train, X_test, y_train, y_test