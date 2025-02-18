# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from tensorflow import keras
# from hyperopt import fmin, tpe, hp, Trials
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error

# def objective(params, X, y, seq_length):
#     """Fonction objective pour Hyperopt utilisant K-Fold."""
#     kf = KFold(n_splits=5, shuffle=False)
#     scores = []
#     for train_index, val_index in kf.split(X):
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]

#         model = Sequential()
#         model.add(LSTM(int(params['units']), activation='relu', return_sequences=True,
#                        input_shape=(seq_length, X.shape[2])))
#         model.add(Dropout(params['dropout']))
#         model.add(LSTM(int(params['units'])//2, activation='relu', return_sequences=False))
#         model.add(Dense(1))
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mse')
#         model.fit(X_train, y_train, epochs=int(params['epochs']),
#                   batch_size=int(params['batch_size']), verbose=0, shuffle=False)
#         y_pred = model.predict(X_val)
#         score = mean_squared_error(y_val, y_pred)
#         scores.append(score)
#     return np.mean(scores)

# def optimize_hyperparameters(X, y, seq_length):
#     """Optimise les hyperparamètres avec Hyperopt."""
#     search_space = {
#         'units': hp.quniform('units', 48, 80, 8),
#         'dropout': hp.uniform('dropout', 0.1, 0.15),
#         'learning_rate': hp.loguniform('learning_rate', np.log(0.0002), np.log(0.0005)),  
#         'epochs': hp.quniform('epochs', 25, 35, 5),       
#         'batch_size': hp.quniform('batch_size', 16, 32, 8)
#     }
#     trials = Trials()
#     # Réduire le nombre d'évaluations
#     best = fmin(fn=lambda params: objective(params, X, y, seq_length),
#                 space=search_space, algo=tpe.suggest, max_evals=5, trials=trials)
#     return best

# def train_best_model(X, y, seq_length, best_params):
#     """Construit et entraîne le modèle LSTM avec les meilleurs hyperparamètres."""
#     model = Sequential()
#     model.add(LSTM(int(best_params['units']), activation='relu', return_sequences=True,
#                    input_shape=(seq_length, X.shape[2])))
#     model.add(Dropout(best_params['dropout']))
#     model.add(LSTM(int(best_params['units'])//2, activation='relu', return_sequences=False))
#     model.add(Dense(1))
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss='mse')
#     model.fit(X, y, epochs=int(best_params['epochs']),
#               batch_size=int(best_params['batch_size']), verbose=1, shuffle=False)
#     return model


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow import keras
from hyperopt import hp, tpe, fmin, Trials
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def objective(params, X, y, seq_length):
    """Fonction objective pour Hyperopt utilisant K-Fold."""
    kf = KFold(n_splits=5, shuffle=False)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = Sequential()
        model.add(LSTM(int(params['units']),
                       activation='relu',
                       return_sequences=True,
                       input_shape=(seq_length, X.shape[2])))
        model.add(Dropout(params['dropout']))
        model.add(LSTM(int(params['units']) // 2,
                       activation='relu',
                       return_sequences=False))
        model.add(Dense(1))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss='mse')
        model.fit(X_train, y_train,
                  epochs=int(params['epochs']),
                  batch_size=int(params['batch_size']),
                  verbose=0,
                  shuffle=False)
        y_pred = model.predict(X_val)
        score = mean_squared_error(y_val, y_pred)
        scores.append(score)
    return np.mean(scores)


def optimize_hyperparameters(X, y, seq_length):
    search_space = {
        'units': hp.quniform('units', 48, 80, 8),
        'dropout': hp.uniform('dropout', 0.1, 0.15),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0002), np.log(0.0005)),
        'epochs': hp.quniform('epochs', 25, 35, 5),
        'batch_size': hp.quniform('batch_size', 16, 32, 8)
    }
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, X, y, seq_length),
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials
    )
    return best


def train_best_model(X, y, seq_length, best_params):
    """Entraîne le modèle LSTM avec les meilleurs hyperparamètres trouvés."""
    model = Sequential()
    model.add(LSTM(int(best_params['units']), activation='relu', return_sequences=True,
                   input_shape=(seq_length, X.shape[2])))
    model.add(Dropout(best_params['dropout']))
    model.add(LSTM(int(best_params['units']) // 2, activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss='mse')
    model.fit(X, y, epochs=int(best_params['epochs']),
              batch_size=int(best_params['batch_size']), verbose=1, shuffle=False)
    return model

def auto_ml(X_train, y_train, X_test, y_test, seq_length, max_evals=5):
    """
    Enchaîne l'optimisation des hyperparamètres et l'entraînement du modèle.
    Retourne le modèle entraîné ainsi que les hyperparamètres optimaux et le score sur le test.
    """
    # On peut combiner X_train et X_test pour l'optimisation ou utiliser uniquement X_train.
    # Ici, nous utilisons X_train pour optimiser, puis nous entraînons sur X_train et évaluons sur X_test.
    best_params = optimize_hyperparameters(X_train, y_train, seq_length)
    print("Meilleurs hyperparamètres trouvés:", best_params)
    model = train_best_model(X_train, y_train, seq_length, best_params)
    # Évaluation sur le set de test
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE sur le set de test: {rmse:.4f}")
    return {"model": model, "best_params": best_params, "rmse": rmse}
