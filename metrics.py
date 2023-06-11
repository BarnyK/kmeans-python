import numpy as np
from typing import Optional, Dict
from itertools import combinations


def silhouette(data: np.ndarray, results: np.ndarray) -> float:
    """Calculates average silhouette coefficient of cluster"""
    no_samples = len(data)
    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)

    # a contains average distances of point i to all other points in its cluster
    a = np.zeros(no_samples)
    for i in range(no_samples):
        cid = results[i]
        selector = results == cid
        selector[i] = False

        # Check if single element cluster
        if selector.sum() == 0:
            a[i] = 0
        else:
            a[i] = np.mean(distances[i, selector])

    # b contains the least average distance of point i to all points
    # of a cluster that does not contain i
    b = np.zeros(no_samples)
    for i in range(no_samples):
        cid = results[i]
        b[i] = np.min(distances[i, results != cid])

    silhouette_coeffs = (b - a) / np.maximum(a, b)
    return np.mean(silhouette_coeffs)


def purity(results: np.ndarray, reference: np.ndarray) -> float:
    """Calculates purity of clustering"""
    no_samples = len(results)

    # Different cluster numberings
    found_clusters = np.unique(results)
    reference_clusters = np.unique(reference)

    # Renumber
    found_renumbering = {y: x for x, y in enumerate(found_clusters)}
    ref_renumbering = {y: x for x, y in enumerate(reference_clusters)}

    # Confusion matrix
    confusion_matrix = np.zeros((len(found_clusters), len(reference_clusters)))
    for i in range(no_samples):
        cid = found_renumbering[results[i]]
        ref_cid = ref_renumbering[reference[i]]
        confusion_matrix[cid][ref_cid] += 1

    # Find max on each row and sum them
    total_correct = np.sum(np.max(confusion_matrix, axis=1))

    return total_correct / no_samples


def rand(results: np.ndarray, reference: np.ndarray) -> float:
    """Calculates the rand index of clustering"""
    n = len(results)
    pairs = list(combinations(range(n), 2))

    # Counting agreements and disagreements
    count = 0
    for i, j in pairs:
        found_same = results[i] == results[j]
        reference_same = reference[i] == reference[j]

        # xnor is true when both values are true or both are false
        # So if both are in the same cluster or both are in different ones
        if not np.logical_xor(found_same, reference_same):
            count += 1

    return count / len(pairs)


def calculate_metrics(
    data: np.ndarray, results: np.ndarray, reference: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculates external and internal clustering qualities"""
    metrics = {}
    if reference is not None:
        metrics["purity"] = purity(results, reference)
        metrics["rand"] = rand(results, reference)
    metrics["silhouette"] = silhouette(data, results)

    return metrics


## simple tests
if __name__ == "__main__":
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    x = np.array([1, 3, 2, 2, 2, 2, 1, 1, 3, 3])
    print(np.array([x, y]).T)

    print(purity(x, y))  # 0.7   Matches slides
    print(rand(x, y))  # 0.51  Matches slides

    x = np.array([1, 1, 2, 1, 2, 2, 3, 3, 3])
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    data = np.array(
        [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5], [0, 5], [0, 6], [1, 5]],
        dtype=float,
    )
    print(silhouette(data, y))  # 0.7421...
