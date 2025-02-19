from typing import Dict
from kedro.pipeline import Pipeline
from .pipeline import create_pipeline
from dotenv import load_dotenv

load_dotenv()


def register_pipelines() -> Dict[str, Pipeline]:
    return {"loading": create_pipeline()}
