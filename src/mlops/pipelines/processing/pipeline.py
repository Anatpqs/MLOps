from kedro.pipeline import Pipeline, node
from .nodes import split_dataset, encode_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                encode_features,
                "primary",
                dict(features="dataset", transform_pipeline="transform_pipeline"),
            ),
            node(
                split_dataset,
                inputs=[
                    "dataset",
                    "params:test_ratio",
                    "params:time_step",
                ],  # Inputs de la node
                outputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    X_val="X_val",
                    y_val="y_val",
                    X_test="X_test",
                    y_test="y_test",
                ),
            ),
        ]
    )
