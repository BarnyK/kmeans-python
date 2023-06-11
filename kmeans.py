import numpy as np
import argparse
from data_handling import load_dataset, save_clustering
from initializations import initialize_centroids
from metrics import calculate_metrics
from helpers import center_equals, assign_labels
from typing import Tuple, Dict

def cluster(data: np.ndarray, centroids: np.ndarray, max_iters: int) -> np.ndarray:
    """Performs the kmeans clustering iteration"""
    current_centroids = centroids.copy()
    for i in range(max_iters):
        new_centroids = _cluster(data, current_centroids)
        if center_equals(current_centroids, new_centroids):
            break
        current_centroids = new_centroids

    return current_centroids, i + 1


def _cluster(data: np.ndarray, current_centroids: np.ndarray) -> np.ndarray:
    """Performs single iteration of kmeans clustering"""
    k = len(current_centroids)

    # Find assignement of each point
    assignement = assign_labels(data, current_centroids)

    # Update centroids by taking a mean of every point assigned to it
    new_centroids = current_centroids.copy()
    for cid in range(k):
        assigned = data[assignement == cid, :]
        # Update mean if points assigned
        if assigned.any():
            new_center = assigned.mean(0)
            new_centroids[cid, :] = new_center

    return new_centroids


def kmeans(
    input_file: str,
    labels_flag: bool,
    k: int,
    init_type: str,
    max_iters: int = 100,
    output_file: str = "",
    verbose: bool = True
) -> Tuple[np.ndarray,np.ndarray,np.ndarray,Dict[str,float],int]:
    """
    Main function performing the kmeans clustering.
    Loads data, initializes centroids, runs clustering iterations, calculates metrics.
    """
    data, labels = load_dataset(input_file, labels_flag)

    if k > data.shape[0]:
        raise ValueError("parameter k can't be bigger than sample size")


    centroids = initialize_centroids(data, k, init_type)

    final_centroids, iterations = cluster(data, centroids, max_iters)

    result = assign_labels(data, final_centroids)

    metrics = calculate_metrics(data, result, labels)

    if verbose:
        if iter == max_iters:
            print("Did not converge in given maximum iterations")
        print(f"Iterations: {iterations}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    if output_file:
        save_clustering(output_file, result)

    return data, final_centroids, result, metrics, iterations


if __name__ == "__main__":
    # filename, labels_flag, k | auto-k
    kmeans("sample.csv", True, 3, "plusplus", 100)
    #kmeans("datasets/artificial/spiral.arff", True, 3, "plusplus", 100)
    #kmeans("datasets/artificial/spiral.arff", True, 3, "random", 100)



if __name__ == "__main__2":
    parser = argparse.ArgumentParser(description="K means clustering")

    # Add arguments
    parser.add_argument(
        "--labels-flag",
        dest="labels_flag",
        action="store_true",
        default=True,
        help="Include labels flag (default: True)",
    )
    parser.add_argument(
        "-k", type=int, required=True, help="Number of clusters (required)"
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="random",
        choices=["random", "plusplus"],
        help="Initialization type (default: random)",
    )
    parser.add_argument(
        "--max-iters", type=int, default=100, help="Maximum iterations (default: 100)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help='Output filename (default: "")',
    )
    parser.add_argument("input_filename", type=str, help="Input filename (required)")

    args = parser.parse_args()
    try:
        kmeans(
            args.input_filename,
            args.labels_flag,
            args.k,
            args.init_type,
            args.max_iters,
            args.out,
        )
    except ValueError as err:
        print(err)
    except FileNotFoundError as err:
        print("File missing")
        print(err)
