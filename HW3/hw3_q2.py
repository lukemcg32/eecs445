from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


class GradientBoostingClassifier445:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, max_depth: int = 3) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_estimator = 0
        self.loss = []  # used to track training loss

    def _sigmoid(self, x: npt.NDArray) -> npt.NDArray:
        """Compute the element-wise sigmoid function for NumPy arrays."""
        return 1 / (1 + np.exp(-x))

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> GradientBoostingClassifier445:
        self.init_estimator = None
        # TODO: initialize base score to the log-odds of the positive class
        self.init_estimator = None

        # initial predictions
        F_m = np.full(y.shape, self.init_estimator)
        self.loss = []
        for _ in range(self.n_estimators):
            model = None
            current_loss = None
            
            # TODO: compute pseudo-residuals
            # TODO: fit a DecisionTreeRegressor to the residuals with random_state=445
            # TODO: update ensemble predictions
            # TODO: calculate the current loss

            # save the model and loss
            self.models.append(model)
            self.loss.append(current_loss)

        return self

    def binary_cross_entropy(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
        """Calculate empirical risk with binary cross-entropy loss."""
        # clip the predictions to ensure we don't take log(0)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        F_m = np.full((X.shape[0],), self.init_estimator)
        for model in self.models:
            F_m += self.learning_rate * model.predict(X)
        return self._sigmoid(F_m)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)


def plot_classifier_performance(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    parameter: str,
    parameters: npt.NDArray,
) -> None:
    classifiers = [GradientBoostingClassifier445(**{parameter: p}).fit(X_train, y_train) for p in tqdm(parameters)]
    train_metrics = [roc_auc_score(y_train, clf.predict_proba(X_train)) for clf in classifiers]
    test_metrics = [roc_auc_score(y_test, clf.predict_proba(X_test)) for clf in classifiers]
    plt.plot(parameters, train_metrics, marker="o", linestyle="-", label="Train")
    plt.plot(parameters, test_metrics, marker="o", linestyle="-", label="Test")
    plt.legend()
    plt.xlabel(parameter)
    plt.ylabel("AUROC")
    plt.title(f"{parameter} vs. AUROC")
    plt.savefig(f"{parameter} vs AUROC.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=200, n_informative=50, n_classes=2, random_state=445)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=445)
    
    # plot the training loss of a classifier
    clf = GradientBoostingClassifier445().fit(X_train, y_train)
    plt.plot(clf.loss, marker="o", linestyle="-")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training Loss")
    plt.savefig("Training Loss.png", bbox_inches="tight")
    plt.show()

    plt.clf()
    # plot the AUROC of the classifier against the number of weak estimators
    plot_classifier_performance(
        X_train,
        y_train,
        X_test,
        y_test,
        "n_estimators",
        np.geomspace(1, 1_000, 7, dtype=int),
    )

    plt.clf()
    # plot the AUROC of the classifier against the max depth of the weak estimators
    plot_classifier_performance(
        X_train,
        y_train,
        X_test,
        y_test,
        "max_depth",
        np.geomspace(1, 25, 10, dtype=int),
    )
