#!/usr/bin/env python
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from google.cloud import storage

# Authentification à Google Cloud avec la clé correspondant au compte de service MLflow
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(
    "conf/local/service_account.json"
)

# Nouvel URI de l'interface MLflow hébergée sur la VM GCP
mlflow.set_tracking_uri("http://35.208.23.119")
# Création du client pour Google Cloud Storage (optionnel si vous souhaitez interagir avec le bucket)
client = storage.Client()


def load_data():
    """
    Charge les datasets depuis les fichiers pickle.
    Les fichiers doivent être présents dans le dossier 'data/05_model_input/'.
    """
    with open("data/05_model_input/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("data/05_model_input/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("data/05_model_input/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open("data/05_model_input/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test


def plot_predictions(test_dates, y_true, y_pred, plot_path):
    """
    Crée un graphique comparant les valeurs réelles et les prédictions,
    puis le sauvegarde sous 'plot_path'.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_true, label="Valeurs réelles")
    plt.plot(test_dates, y_pred, label="Prédictions", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture")
    plt.title("Prévision des prix de clôture par LSTM")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


def main():
    # Configuration de MLflow avec le tracking URI pointant vers le serveur distant
    mlflow.set_experiment("stock_forecasting")

    # Chargement des datasets prétraités
    X_train, X_test, y_train, y_test = load_data()

    # Utilisation des meilleurs hyperparamètres déjà définis
    best_params = {
        "units": 64,
        "dropout": 0.11577068385301886,
        "epochs": 30,
        "batch_size": 16,
        "learning_rate": 0.00031186448298291594,
    }

    # La longueur de la séquence est déterminée par la deuxième dimension de X_train
    seq_length = X_train.shape[
        1
    ]  # X_train de forme (nb_samples, seq_length, nb_features)

    # Pour le graphique, on simule des dates de test à partir d'une date fictive
    test_dates = pd.date_range(start="2020-01-01", periods=len(y_test))

    # Démarrer un run MLflow pour tracker cette expérience
    with mlflow.start_run() as run:
        # Construction du modèle LSTM
        model = Sequential()
        model.add(
            LSTM(
                int(best_params["units"]),
                activation="relu",
                return_sequences=True,
                input_shape=(seq_length, X_train.shape[2]),
            )
        )
        model.add(Dropout(best_params["dropout"]))
        model.add(
            LSTM(
                int(best_params["units"]) // 2,
                activation="relu",
                return_sequences=False,
            )
        )
        model.add(Dense(1))

        optimizer = Adam(learning_rate=best_params["learning_rate"])
        model.compile(optimizer=optimizer, loss="mse")

        # Entraînement du modèle sur les données d'entraînement
        model.fit(
            X_train,
            y_train,
            epochs=int(best_params["epochs"]),
            batch_size=int(best_params["batch_size"]),
            verbose=1,
            shuffle=False,
        )

        # Prédictions sur le jeu de test et calcul de la RMSE
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Log des hyperparamètres et de la métrique dans MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)

        # Création et sauvegarde du graphique des prédictions
        os.makedirs("plots", exist_ok=True)
        plot_path = "plots/prediction_plot.png"
        plot_predictions(test_dates, y_test, y_pred, plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # Inférer la signature du modèle à partir des données d'entraînement
        signature = infer_signature(X_train, model.predict(X_train))
        # Log du modèle entraîné dans MLflow avec mlflow.keras
        mlflow.keras.log_model(model, "lstm_model", signature=signature)

        print(f"Run ID: {run.info.run_id}")
        print(f"Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
