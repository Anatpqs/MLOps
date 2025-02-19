import pandas as pd

from mlops.pipelines.loading.nodes import load_csv_from_bucket

def test_load_csv_from_bucket(project_id, primary_folder):
    df = load_csv_from_bucket(project_id, primary_folder)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 7
    assert "High" in df
