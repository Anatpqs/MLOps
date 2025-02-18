from typing import Dict
from kedro.pipeline import Pipeline
from .pipeline import create_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "loading": create_pipeline()  # Retourne la pipeline `processing`
    }
