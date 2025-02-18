from kedro.pipeline import Pipeline, node
from .nodes import load_data, compute_features, create_sequences_and_scale, split_data

def create_pipeline():
    return Pipeline([
        node(
            func=load_data,
            inputs="params:raw_filepath",  # chemin défini dans le catalog ou paramètres
            outputs="raw_data",
            name="load_data_node"
        ),
        node(
            func=compute_features,
            inputs="raw_data",
            outputs="processed_data",
            name="compute_features_node"
        ),
        node(
            func=create_sequences_and_scale,
            inputs=dict(df="processed_data", features="params:features", seq_length="params:seq_length"),
            outputs=["X", "y", "scaler"],
            name="create_sequences_node"
        ),
        node(
            func=split_data,
            inputs=dict(X="X", y="y", test_ratio="params:test_ratio"),
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),
    ])
