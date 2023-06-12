import numpy as np


def random_initialization(data: np.ndarray, k: int) -> np.ndarray:
    """Choose random points"""
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def plusplus_initialization(data: np.ndarray, k: int) -> np.ndarray:
    """
    Kmeans++ initialization
    First point is random
    Then each new point is chosen based on distance to already chosen centroids
    """
    centroids = data[np.random.randint(0, data.shape[0])]
    centroids = np.expand_dims(centroids, axis=0)
    expanded_data = np.expand_dims(data, axis=1)

    for _ in range(1, k):
        distances = np.sqrt(np.sum((expanded_data - centroids) ** 2, axis=2))
        distances = np.min(distances, 1)
        probabilities = distances / distances.sum()
        new_index = np.random.choice(data.shape[0], p=probabilities)
        centroids = np.append(centroids, data[new_index : new_index + 1, :], axis=0)

    return centroids


def initialize_centroids(data: np.ndarray, k: int, init_type: str) -> np.ndarray:
    """
    Initialize centroids
    """
    if init_type == "random":
        return random_initialization(data, k)
    elif init_type == "plusplus":
        return plusplus_initialization(data, k)
    else:
        raise ValueError(f"No initialization for init_type={init_type!r}")
