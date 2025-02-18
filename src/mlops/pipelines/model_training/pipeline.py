from kedro.pipeline import Pipeline, node
from .nodes import auto_ml

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=auto_ml,
            inputs=["X_train", "y_train", "X_test", "y_test", "params:seq_length", "params:automl_max_evals"],
            outputs="trained_model",
            name="auto_ml_node"
        )
    ])
