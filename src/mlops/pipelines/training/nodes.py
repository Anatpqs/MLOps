import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from hyperopt import hp, tpe, fmin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import warnings

warnings.filterwarnings("ignore")

# Définir le modèle LSTM et ses hyperparamètres
MODELS = [
    {
        "name": "LSTM",
        "class": "LSTM",
        "params": {
            "units": hp.quniform("units", 32, 512, 32),
            "dropout_rate": hp.uniform("dropout_rate", 0.2, 0.5),
            "learning_rate": hp.uniform("learning_rate", 0.001, 0.01),
            "batch_size": hp.quniform("batch_size", 16, 128, 16),
            "epochs": hp.quniform("epochs", 10, 100, 10),
        },
    }
]


def create_lstm_model(input_shape, units, dropout_rate, learning_rate):
    """
    Création d'un modèle LSTM simple avec Keras.
    """
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))  # Pour classification binaire

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    instance: str,
    training_set: Tuple[np.ndarray, np.ndarray],
    params: Dict = {},
) -> tf.keras.Model:
    """
    Entraîne une instance du modèle LSTM avec les données fournies et les hyperparamètres.
    """
    X_train, y_train = training_set
    # Reshaping les données pour LSTM (format attendu par LSTM: [samples, timesteps, features])
    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1], 1
    )  # Ajouter une dimension pour features

    # Créer le modèle LSTM
    model = create_lstm_model(
        input_shape=(X_train.shape[1], 1),  # Nombre de timesteps et features
        units=params["units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"],
    )

    # Entraîner le modèle
    model.fit(
        X_train,
        y_train,
        batch_size=int(params["batch_size"]),
        epochs=int(params["epochs"]),
        verbose=0,
    )
    return model


def optimize_hyp(
    dataset: Tuple[np.ndarray, np.ndarray],
    search_space: Dict,
    metric: Callable[[Any, Any], float],
    max_evals: int = 40,
) -> Dict:
    """
    Optimisation des hyperparamètres pour un modèle LSTM.
    """
    X, y = dataset

    def objective(params):
        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1)
        scores_test = []
        for train_I, test_I in rep_kfold.split(X):
            X_fold_train = X[train_I]
            y_fold_train = y[train_I]
            X_fold_test = X[test_I]
            y_fold_test = y[test_I]

            # Entraîner le modèle LSTM avec les hyperparamètres
            model = train_model_lstm(
                instance="LSTM",
                training_set=(X_fold_train, y_fold_train),
                params=params,
            )

            # Calculer le score (ici, on utilise F1)
            scores_test.append(
                metric(y_fold_test, (model.predict(X_fold_test) > 0.5).astype(int))
            )

        return np.mean(scores_test)

    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals)


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40,
) -> Dict:
    """
    Entraîne plusieurs modèles LSTM et sélectionne le plus performant basé sur F1-score.
    """
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Définir les hyperparamètres à tester pour le modèle LSTM
    opt_models = []
    model_specs = MODELS[0]  # Seulement le LSTM pour le moment
    optimum_params = optimize_hyp_lstm(
        dataset=(X, y),
        search_space=model_specs["params"],
        metric=lambda x, y: -f1_score(x, y),
        max_evals=max_evals,
    )
    print("done")

    # Entraînement du meilleur modèle avec les hyperparamètres trouvés
    model = train_model_lstm(
        instance="LSTM", training_set=(X_train, y_train), params=optimum_params
    )

    # Calculer le score F1 sur le test
    score = f1_score(y_test, (model.predict(X_test) > 0.5).astype(int))

    opt_models.append(
        {
            "model": model,
            "name": model_specs["name"],
            "params": optimum_params,
            "score": score,
        }
    )

    best_model = max(opt_models, key=lambda x: x["score"])
    return dict(model=best_model)
