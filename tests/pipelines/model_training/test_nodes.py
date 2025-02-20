import numpy as np
from src.mlops.pipelines.model_training.nodes import objective, train_best_model


def test_objective():
    """Unit test for the optimization objective function."""
    X_dummy = np.random.rand(100, 30, 5)
    y_dummy = np.random.rand(100)
    seq_length = 30
    params = {
        "units": 64,
        "dropout": 0.1,
        "learning_rate": 0.0003,
        "epochs": 5,
        "batch_size": 16,
    }
    score = objective(params, X_dummy, y_dummy, seq_length)
    assert isinstance(score, float), "The returned score must be a float"
    assert score >= 0, "The MSE metric cannot be negative"


def test_train_best_model():
    """Unit test to verify that the trained model is a keras.Model object."""
    X_dummy = np.random.rand(100, 30, 5)
    y_dummy = np.random.rand(100)
    seq_length = 30
    best_params = {
        "units": 64,
        "dropout": 0.1,
        "learning_rate": 0.0003,
        "epochs": 5,
        "batch_size": 16,
    }
    model = train_best_model(X_dummy, y_dummy, seq_length, best_params)
    assert model is not None, "The model must not be None"
    assert hasattr(model, "predict"), "The returned object must have a predict method"


def test_model_prediction():
    """Directional test to check if increasing a feature correctly influences the output."""
    X_dummy = np.random.rand(100, 30, 5)
    y_dummy = np.random.rand(100)
    seq_length = 30
    best_params = {
        "units": 64,
        "dropout": 0.1,
        "learning_rate": 0.0003,
        "epochs": 5,
        "batch_size": 16,
    }
    model = train_best_model(X_dummy, y_dummy, seq_length, best_params)
    X_modified = X_dummy.copy()
    X_modified[:, :, 0] += 0.1  # Artificially increase a feature
    y_pred_original = model.predict(X_dummy)
    y_pred_modified = model.predict(X_modified)
    assert np.mean(y_pred_modified) != np.mean(
        y_pred_original
    ), "The model must react to a modification of the features"


def test_invariance():
    """Invariance test to check that shuffling the data does not significantly modify predictions."""
    X_dummy = np.random.rand(100, 30, 5)
    y_dummy = np.random.rand(100)
    seq_length = 30
    best_params = {
        "units": 64,
        "dropout": 0.1,
        "learning_rate": 0.0003,
        "epochs": 5,
        "batch_size": 16,
    }
    model = train_best_model(X_dummy, y_dummy, seq_length, best_params)
    y_pred_original = model.predict(X_dummy)
    X_shuffled = np.random.permutation(X_dummy)
    y_pred_shuffled = model.predict(X_shuffled)
    diff = np.mean(np.abs(y_pred_original - y_pred_shuffled))
    assert diff < 0.1, f"Predictions change after permutation (diff: {diff})"
