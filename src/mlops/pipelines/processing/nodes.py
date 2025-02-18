import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler


# Préparer les données pour LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data.iloc[i : (i + time_step), 0].values  # Séquence de taille time_step
        X.append(a)
        y.append(data.iloc[i + time_step, 0])  # Cible suivante
    return np.array(X), np.array(y)


def encode_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features of Yahoo Finance dataset without time-based features.
    """
    # Sélectionner les colonnes pertinentes
    features = dataset.copy()

    # Normalisation des données numériques
    scaler = MinMaxScaler()
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    features[numeric_columns] = scaler.fit_transform(features[numeric_columns])

    return dict(features=features, transform_pipeline=scaler)


# Fonction pour diviser le dataset


def split_dataset(
    dataset: pd.DataFrame, test_ratio: float, time_step: int = 60
) -> Dict[str, Any]:
    """
    Divise le dataset en ensemble d'entraînement, validation et test pour LSTM.
    Retourne les données sous forme de pandas DataFrame, prêtes à être enregistrées en CSV.
    """

    # Split dataset en training et testing
    train_size = int(
        len(dataset) * (1 - test_ratio)
    )  # Utiliser test_ratio pour ajuster la taille du train
    val_size = int(len(dataset) * 0.1)  # 10% pour la validation
    train_data = dataset[: train_size - val_size]
    val_data = dataset[train_size - val_size : train_size]
    test_data = dataset[train_size:]

    # Création des datasets pour LSTM avec la fonction create_dataset
    X_train, y_train = create_dataset(train_data, time_step)
    X_val, y_val = create_dataset(val_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape des données pour LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Convertir les numpy arrays en DataFrames pandas
    X_train_df = pd.DataFrame(
        X_train.reshape(X_train.shape[0], -1)
    )  # Aplatir pour chaque feature
    X_val_df = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
    X_test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))

    # Convertir les cibles (y_train, y_val, y_test) en DataFrames
    y_train_df = pd.DataFrame(y_train)
    y_val_df = pd.DataFrame(y_val)
    y_test_df = pd.DataFrame(y_test)

    # Retourner sous forme de DataFrame pour chaque ensemble
    return {
        "X_train": X_train_df,
        "y_train": y_train_df,
        "X_val": X_val_df,
        "y_val": y_val_df,
        "X_test": X_test_df,
        "y_test": y_test_df,
    }
