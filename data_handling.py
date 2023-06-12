import numpy as np
import arff
from typing import Tuple
import csv
from os import path


def load_arff(filename: str, labels_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset from arff
    """
    dataset = arff.load(open(filename, "r"))
    if labels_flag:
        data = np.array([x[:-1] for x in dataset["data"]], dtype=float)
        labels = np.array([x[-1] for x in dataset["data"]], dtype=int)
    else:
        data = np.array([x for x in dataset["data"]], dtype=float)
        labels = None
    return data, labels


def load_csv(filename: str, labels_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset from csv
    """
    with open(filename) as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        array = [x for x in reader]
    if labels_flag:
        data = np.array([x[:-1] for x in array], dtype=float)
        labels = np.array([x[-1] for x in array], dtype=int)
    else:
        data = np.array(array, dtype=float)
        labels = None
    return data, labels


def load_dataset(filename: str, labels_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from a file
    labels flag controls if last row of the data is a label
    """
    if not path.isfile(filename):
        raise FileNotFoundError(f"{filename} does not exist")
    ext = path.splitext(filename)[-1]
    if ext == ".arff":
        data, labels = load_arff(filename, labels_flag)
    elif ext == ".csv":
        data, labels = load_csv(filename, labels_flag)
    else:
        raise ValueError("File of incorrect format")
    return data, labels


def save_arff(filename: str, clustering: np.ndarray):
    """
    Writes arff file with single containing cluster ids
    """
    data = np.transpose([clustering])
    attributes = [("cluster_label", "NUMERIC")]
    arff_dict = {"relation": "clustering", "attributes": attributes, "data": data}
    with open(filename, "w") as f:
        arff.dump(arff_dict, f)


def save_csv(filename: str, clustering: np.ndarray):
    """
    Writes csv file with single containing cluster ids
    """
    data = np.transpose([clustering])
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["clustering"])
        writer.writerows(data.tolist())


def save_clustering(filename: str, clustering: np.ndarray):
    """
    Saves clustering to a file
    """
    ext = path.splitext(filename)[-1]
    if ext == ".arff":
        save_arff(filename, clustering)
    elif ext == ".csv":
        save_csv(filename, clustering)
    else:
        raise ValueError("Unknown output file format")
