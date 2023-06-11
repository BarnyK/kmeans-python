import numpy as np


def calculate_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates euclidean distance between elements of 2 arrays"""
    return np.linalg.norm(x[:, np.newaxis] - y, axis=2)


def center_equals(x: np.ndarray, y: np.ndarray) -> bool:
    """Compares 2 centroid matrices"""
    distances = np.sqrt(np.sum((x - y) ** 2, 1))
    if np.any(distances > 1e-10):  # Accuracy to be tested
        return False
    return True


def assign_labels(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Calculates distances between data points and centroids
    and finds the closest for each one.
    """
    distances = calculate_distances(data, centroids)
    closest = np.argmin(distances, axis=1)
    return closest