import numpy as np
from sklearn.cluster import KMeans


def get_k_representatives(pixels: np.array, k: int = 5) -> list:
    """Given an array (N, 3), compute the k cluster and then return the centroids
    which are representatives of the pixels group given."""
    k_means_cluster = KMeans(n_clusters=k, n_init=5).fit(pixels)
    centroids = k_means_cluster.cluster_centers_

    # Sort the centroids by their proximity to the origin
    centroids_distances = np.linalg.norm(centroids, axis=1)
    sorted_centroids_indices = np.argsort(centroids_distances)
    sorted_centroids = centroids[sorted_centroids_indices]

    return sorted_centroids