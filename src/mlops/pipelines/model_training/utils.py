import matplotlib.pyplot as plt


def save_pr_curve(X, y, model):

    y_pred = model.predict(X)
    plt.figure(figsize=(16, 11))
    plt.plot(y, label="Valeurs réelles")
    plt.plot(y_pred, label="Prédictions", linestyle="--")
    plt.title("Courbe de régression", fontsize=16)
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.legend()
    plt.savefig("data/08_reporting/pr_curve.png")
    plt.close()
