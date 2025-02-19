import pytest
import pandas as pd
from src.mlops.pipelines.loading.nodes import load_csv_from_bucket
from src.mlops.pipelines.data_processing.nodes import compute_features
from kedro.io import DataCatalog, MemoryDataset

@pytest.fixture(scope="module")
def project_id():
    return "mlops-451309"

@pytest.fixture(scope="module")
def primary_folder():
    return "bucket_projet_mlops/primary/primary.csv"
    
@pytest.fixture(scope="module")
def dataset_not_encoded(project_id, primary_folder):
    return load_csv_from_bucket(project_id, primary_folder)

@pytest.fixture(scope="module")
def test_ratio():
    return 0.3

@pytest.fixture(scope="module")
def dataset_encoded(dataset_not_encoded):
    return compute_features(dataset_not_encoded)

# Fixture pour créer le DataCatalog avec les datasets nécessaires
@pytest.fixture(scope="module")
def catalog_test(dataset_not_encoded, dataset_encoded, test_ratio):
    # Créer un DataCatalog avec les datasets nécessaires
    features_list = ["feature1", "feature2"]
    
    catalog = DataCatalog({
        "primary": MemoryDataset(dataset_not_encoded),
        "encoded_data": MemoryDataset(dataset_encoded),
        "params:test_ratio": MemoryDataset(test_ratio),
        "params:features": MemoryDataset(features_list),  # Remplace par la valeur réelle attendue
        "params:seq_length": MemoryDataset(100),  # Remplace par la valeur réelle attendue
        "params:raw_filepath": MemoryDataset("data/01_raw/yahoo_stock.csv"),
    })
    return catalog