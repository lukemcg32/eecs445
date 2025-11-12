import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans

PLT_MARKERS = ["o", "s", "D", "^", "v", "p", "*", "X"]


def visualize_kmeans(data: npt.NDArray, n_clusters: int, init: str) -> None:
    """
    Trains a k-means clustering model on the given data and visualizes the resulting clusters.

    Args:
        data: A 2D NumPy array of shape (n_samples, n_features) representing the input data to be clustered.
        n_clusters: The number of clusters to form.
        init: The initialization method for the centers.

    Returns:
        The trained scikit-learn KMeans object.
    """

    # TODO: Define a scikit-learn KMeans object
    # - Set argument n_clusters (number of clusters) to n_clusters
    # - Set argument init ('random' or 'k-means++') to init
    # - Set random_state to 95
    # - Set n_init to 10
    kmeans = None

    # Fit data to obtain clusters
    kmeans.fit(data)

    # TODO: Print final value of objective function

    # Plot each cluster on the same axes
    plt.figure()
    for cluster in np.arange(n_clusters):
        plt.plot(data[kmeans.labels_ == cluster, 0], data[kmeans.labels_ == cluster, 1], PLT_MARKERS[cluster])
    plt.title(f"K-Means Clustering Visualization - {init}")
    plt.savefig(f"kmeans_visualization_{init}.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved to kmeans_visualization_{init}.png")


def plot_inertia(X: npt.NDArray, init: str) -> None:
    """
    Plot inertias over a range of number of clusters.
    """

    n_clusters = np.arange(2, 13)
    inertias = []
    for k in n_clusters:
        # Keep the random state at 95 for clearer results
        clf = KMeans(k, init=init, random_state=95).fit(X)
        # TODO: Add the inertia for each fit classifier to the list of inertias

    plt.figure()
    plt.plot(n_clusters, inertias, linestyle="-")
    plt.scatter(n_clusters, inertias, marker="o")
    plt.xlabel("Number of Clusters ($k$)")
    plt.ylabel("Inertia")
    plt.savefig("inertias.png", bbox_inches="tight")
    print(f"Plot saved to inertias.png")


if __name__ == "__main__":
    X = pd.read_csv("q5_data/data.csv").to_numpy()

    # TODO: Implement visualize_kmeans and run the script with the appropriate init value above

    # TODO: Implement plot_inertia to generate a plot of KMeans losses with different numbers of clusters
    plot_inertia(X, "k-means++")
