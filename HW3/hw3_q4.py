import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # create the dataset
    np.random.seed(445)
    X = np.empty((100, 2))
    X[:50, 0] = np.full(50, -1)
    X[50:, 0] = np.full(50, 1)
    X[:50, 1] = np.random.normal(loc=50, scale=10, size=50)
    X[50:, 1] = np.random.normal(loc=50, scale=10, size=50)

    # visualize KMeans clustering without standardization
    kmeans = KMeans(n_clusters=2, init="k-means++", random_state=445).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.show()

    # TODO: visualize KMeans clustering with standardization
    raise NotImplementedError()
