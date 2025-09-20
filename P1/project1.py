"""
EECS 445 Fall 2025

This script contains most of the work for the project. You will need to fill in every TODO comment.
"""

import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import helper

__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)


def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: DataFrame with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df = df.copy()
    df["Value"] = df["Value"].replace(-1, np.nan).astype(float)

    feature_dict: dict[str, float] = {}

    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for var in static_variables:
        vals = df.loc[df["Variable"] == var, "Value"].dropna()
        feature_dict[var] = float(vals.iloc[0]) if not vals.empty else np.nan

    # TODO: 3) extract max of time-varying variables into feature dict
    ts = df[df["Variable"].isin(timeseries_variables)]
    max_by_var = ts.groupby("Variable")["Value"].max()

    for var in timeseries_variables:
        feature_dict[f"max_{var}"] = float(max_by_var.get(var, np.nan))

    return feature_dict


def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: (n, d) feature matrix, which could contain missing values

    Returns:
        X: (n, d) feature matrix, without missing values
    """
    # raise NotImplementedError()  
    # TODO: implement

    df = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    col_means = df.mean(axis=0, skipna=True).fillna(0.0)
    df = df.fillna(col_means)
    return df.to_numpy(dtype=float)
    


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (n, d) feature matrix

    Returns:
        X: (n, d) feature matrix with values that are normalized per column
    """
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    # raise NotImplementedError()  
    # TODO: implement
    X = X.astype(float) #make sure we're doing float?

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm

    


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: The name of the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel: The name of the Kernel used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(penalty=penalty, 
                                      C=C, 
                                      solver="liblinear" if penalty == "l1" else "lbfgs",
                                      fit_intercept=False,
                                      class_weight=class_weight,
                                      random_state=seed)
    elif loss == "squared_error":
            return KernelRidge(alpha=1.0 / C, kernel=kernel, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function: {loss}")


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy",
    bootstrap: bool = False,
) -> float | tuple[float, float, float]:
    """
    Calculates the performance metric as evaluated on the true labels y_true versus the predicted scores from
    clf_trained and X. Returns single sample performance if bootstrap is False, otherwise returns the median
    and the empirical 95% confidence interval. You may want to implement an additional helper function to
    reduce code redundancy.

    Args:
        clf_trained: a fitted sklearn estimator
        X: (n, d) feature matrix
        y_true: (n, ) vector of labels in {+1, -1}
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1_score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
        bootstrap: whether to use bootstrap sampling for performance estimation
    
    Returns:
        If bootstrap is False, returns the performance for the specific metric. If bootstrap is True, returns
        the median and the empirical 95% confidence interval.
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    
    # helper to compute our score once to limit redundancy
    def compute_once(X_local: np.ndarray, y_local: np.ndarray) -> float:
        if hasattr(clf_trained, "decision_function"):
            scores = clf_trained.decision_function(X_local)
        else:
            scores = clf_trained.predict(X_local)

        # hard labels for discrete metrics
        y_pred = np.where(scores >= 0.0, 1, -1)

        if metric == "accuracy":
            return float(accuracy_score(y_local, y_pred))
        elif metric == "precision":
            # 1: "positive", if zero div happens make it 0
            return float(precision_score(y_local, y_pred, pos_label=1, zero_division=0))
        elif metric in {"f1", "f1_score"}:
            return float(f1_score(y_local, y_pred, pos_label=1, zero_division=0))
        elif metric == "auroc":
            return float(roc_auc_score(y_local, scores))
        elif metric == "average_precision":
            return float(average_precision_score(y_local, scores, pos_label=1))
        elif metric == "sensitivity":
            cm = confusion_matrix(y_local, y_pred, labels=[-1, 1])
            fn = cm[1][0]
            tp = cm[1][1]

            # make sure we don't do zero division
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        elif metric == "specificity":
            cm = confusion_matrix(y_local, y_pred, labels=[-1, 1])
            tn = cm[0][0]
            fp = cm[0][1]

            # make sure we don't do zero division
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            # hopefully we don't have to ever see this
            import sys
            print("Error: unrecognized metric", file=sys.stderr)
            sys.exit(1)   # non-zero means error

    if not bootstrap:
        # compute return non-bootstrapped one time go
        return compute_once(X, y_true)

    rng = np.random.default_rng(seed)
    B = 1000
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        vals.append(compute_once(X[idx], y_true[idx]))
    lo, med, hi = np.percentile(vals, [2.5, 50.0, 97.5])
    return float(med), float(lo), float(hi)








def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
        k: the number of folds
        metric: the performance metric (default="accuracy"
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    # NOTE: you may find sklearn.model_selection.StratifiedKFold helpful
    # raise NotImplementedError()  # TODO: implement
    # sklearn.linear model.LogisticRegression

    skf = StratifiedKFold(n_splits=k, shuffle=False) # for grading dont shuffle
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        classifier = clf
        classifier.fit(X_tr, y_tr)

        s = performance(classifier, X_va, y_va, metric=metric)  # no bootstrapping by default
        scores.append(s)

    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))




def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of true labels in {+1, -1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision", "sensitivity",
                and "specificity")
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # NOTE: use your cv_performance function to evaluate the performance of each classifier
    # raise NotImplementedError()  # TODO: implement
    best_score = -np.inf
    best_C = C_range[0]
    best_penalty = penalties[0]


    for penalty in penalties:
        for C in C_range:
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=seed)
            mean_score, min_score, max_score = cv_performance(clf, X, y, metric=metric, k=k)

            # best score, or smallest C
            if (mean_score > best_score) or (round(mean_score, 6) == round(best_score, 6) and C < best_C):
                best_score = mean_score
                best_C = C
                best_penalty = penalty
    
    return best_C, best_penalty









def select_param_RBF(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    gamma_range: list[float],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of binary labels {1, -1}
        k: the number of folds 
        metric: the performance metric (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # NOTE: this function should be similar to your implementation of select_param_logreg
    raise NotImplementedError()  # TODO: implement


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    
    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # TODO: initialize clf with C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=seed)
            
            # TODO: fit clf to X and y
            clf.fit(X, y)
            
            # TODO: extract learned coefficients from clf into w
            # NOTE: the sklearn.linear_model.LogisticRegression documentation will be helpful here
            w = clf.coef_
            
            # TODO: count the number of nonzero coefficients and append the count to norm0
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: If you're having issues loading the data (e.g. your computer crashes, runs out of memory,
    #       debug statements aren't printing correctly, etc.) try setting n_jobs = 1 in get_project_data.
    X_train, y_train, X_test, y_test, feature_names = helper.get_project_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!

    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       helper.save_challenge_predictions to save your predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()


if __name__ == "__main__":
    main()
