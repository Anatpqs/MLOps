from kedro.runner import SequentialRunner
from mlops.pipelines.model_training.pipeline import create_pipeline


def test_pipeline(catalog_test):
    """Integration test for the Kedro pipeline with separate outputs."""

    runner = SequentialRunner()
    pipeline = create_pipeline()
    pipeline_output = runner.run(pipeline, catalog_test)

    # Vérifier que le pipeline produit les 4 sorties attendues
    assert "model" in pipeline_output, "The pipeline does not generate 'model' output."
    assert (
        "best_params" in pipeline_output
    ), "The pipeline does not generate 'best_params' output."
    assert "rmse" in pipeline_output, "The pipeline does not generate 'rmse' output."
    assert (
        "mlflow_run_id" in pipeline_output
    ), "The pipeline does not generate 'mlflow_run_id' output."

    model = pipeline_output["model"]
    best_params = pipeline_output["best_params"]
    rmse = pipeline_output["rmse"]
    mlflow_run_id = pipeline_output["mlflow_run_id"]

    # Verify that the model is a Keras model
    assert hasattr(model, "predict"), "The trained model must have a predict method."

    # Verify that the best_params is a dictionary
    assert isinstance(best_params, dict), "The hyperparameters must be a dictionary."

    # Verify that the RMSE is a float
    assert isinstance(rmse, float), "The RMSE must be a float."

    # Verify that the mlflow_run_id is a string
    assert isinstance(mlflow_run_id, str), "mlflow_run_id should be a string."
