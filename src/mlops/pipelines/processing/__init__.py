from typing import Dict
from kedro.pipeline import Pipeline
from .pipeline import create_pipeline  # Importation de la fonction create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Expose la pipeline `processing`.
    """
    return {"processing": create_pipeline()}  # Retourne la pipeline `processing`
