import pandas as pd
import numpy as np

from src.mlops.pipelines.data_processing.nodes import compute_features
from src.mlops.pipelines.data_processing.nodes import split_data

def test_compute_features(dataset_not_encoded):
    df = compute_features(dataset_not_encoded)

    # Vérifier que les colonnes attendues sont bien créées
    assert "MA7" in df.columns
    assert "MA30" in df.columns
    assert "Return" in df.columns
    assert "feature1" in df.columns
    assert "feature2" in df.columns

    # Check if any NaN values remain after dropping rows
    assert df.isna().sum().sum() == 0, "There are still NaN values in the dataframe"

    # Check that `feature1` is numeric
    assert pd.api.types.is_numeric_dtype(df["feature1"]), "'feature1' should be numeric"
    
    # Check that `feature2` is numeric
    assert pd.api.types.is_numeric_dtype(df["feature2"]), "'feature2' should be numeric"

def test_split_data(dataset_encoded, test_ratio):
    X = dataset_encoded.drop(columns=["Close"])  # Garde toutes les colonnes sauf la cible
    y = dataset_encoded["Close"]  # La variable cible (prix de clôture)

    X_train, X_test, y_train, y_test = split_data(X, y, test_ratio)  # ⚠️ PAS .values() !

    # Vérifications
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == dataset_encoded.shape[0]
    
    assert np.ceil(dataset_encoded.shape[0] * test_ratio) == X_test.shape[0]
