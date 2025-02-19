from dotenv import load_dotenv
from typing import Dict
from kedro.pipeline import Pipeline
from .pipeline import create_pipeline

load_dotenv()


def register_pipelines() -> Dict[str, Pipeline]:
    return {"deployment": create_pipeline()}
