from kedro.runner import SequentialRunner
from mlops.pipelines.model_training.pipeline import create_pipeline


def test_pipeline(catalog_test):
    """Integration test for the Kedro pipeline."""
    runner = SequentialRunner()
    pipeline = create_pipeline()
    pipeline_output = runner.run(pipeline, catalog_test)

    assert (
        "trained_model" in pipeline_output
    ), "The pipeline does not generate the expected output."

    trained_model_output = pipeline_output["trained_model"]
    assert isinstance(
        trained_model_output, dict
    ), "The model output must be a dictionary."

    assert "model" in trained_model_output, "The output must contain the trained model."
    assert hasattr(
        trained_model_output["model"], "predict"
    ), "The trained model must have a predict method."

    assert (
        "best_params" in trained_model_output
    ), "The output must contain the best hyperparameters."
    assert isinstance(
        trained_model_output["best_params"], dict
    ), "The hyperparameters must be a dictionary."

    assert "rmse" in trained_model_output, "The output must contain the RMSE metric."
    assert isinstance(trained_model_output["rmse"], float), "The RMSE must be a float."
