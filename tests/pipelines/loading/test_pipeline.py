from kedro.runner import SequentialRunner
from mlops.pipelines.loading.pipeline import create_pipeline
import pandas as pd


def test_pipeline(catalog_test):
    runner = SequentialRunner()
    pipeline = create_pipeline()
    pipeline_output = runner.run(pipeline, catalog_test)
    df = pipeline_output["primary"]
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 7
