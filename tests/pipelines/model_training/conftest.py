import pytest
import os
import joblib
from kedro.io import DataCatalog, MemoryDataset


@pytest.fixture(scope="session")
def catalog_test():
    """Creates an in-memory Kedro DataCatalog for testing."""
    return DataCatalog(
        {
            "X_train": MemoryDataset(
                joblib.load(os.path.expanduser("./data/05_model_input/X_train.pkl"))
            ),
            "y_train": MemoryDataset(
                joblib.load(os.path.expanduser("./data/05_model_input/y_train.pkl"))
            ),
            "X_test": MemoryDataset(
                joblib.load(os.path.expanduser("./data/05_model_input/X_test.pkl"))
            ),
            "y_test": MemoryDataset(
                joblib.load(os.path.expanduser("./data/05_model_input/y_test.pkl"))
            ),
            "model": MemoryDataset(
                joblib.load(os.path.expanduser("./data/06_models/trained_model.pkl"))
            ),
            "params:seq_length": MemoryDataset(30),
            "params:automl_max_evals": MemoryDataset(1),
            "params:mlflow_enabled": MemoryDataset(False),  # ou True, selon ton besoin
            "params:mlflow_experiment_id": MemoryDataset(-1),  # ou 1, selon ton besoin
        }
    )


@pytest.fixture(scope="module")
def model(catalog_test):
    """Loads the trained model from the catalog."""
    return catalog_test.load("model")


@pytest.fixture(scope="module")
def X_test(catalog_test):
    """Loads the test data from the catalog."""
    return catalog_test.load("X_test")


@pytest.fixture(scope="module")
def y_test(catalog_test):
    """Loads the test labels from the catalog."""
    return catalog_test.load("y_test")


@pytest.fixture(scope="module")
def shap_explainer(model):
    """Creates a SHAP explainer for the model."""
    import shap

    return shap.TreeExplainer(model)


@pytest.fixture(scope="module")
def shap_values(shap_explainer, X_test):
    """Computes SHAP values for the test data."""
    return shap_explainer.shap_values(X_test)[1]
