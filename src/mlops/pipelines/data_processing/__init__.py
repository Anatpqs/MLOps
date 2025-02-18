from typing import Dict
from kedro.pipeline import Pipeline
from .pipeline import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Registers the pipeline and returns a dictionary of pipelines."""
    return {"loading": create_pipeline()}  # Returns the `data_processing` pipeline
