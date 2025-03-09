import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from config import CENTERS, CLUSTER_STD, N_SAMPLES, RANDOM_STATE


def generate_dataset():
    """
    Generates a synthetic dataset using Gaussian blobs.

    Returns:
        X (ndarray): Feature matrix.
        Y (ndarray): Label vector.
    """
    X, Y = make_blobs(
        n_samples=N_SAMPLES,
        centers=CENTERS,
        cluster_std=CLUSTER_STD,
        random_state=RANDOM_STATE,
    )
    return X, Y


def split_dataset(X, Y, test_size=0.2):
    """
    Splits the dataset into training and test sets.

    Args:
        X (ndarray): Feature matrix.
        Y (ndarray): Label vector.
        test_size (float): Fraction of dataset for testing.

    Returns:
        X_train, X_test, Y_train, Y_test: Split datasets.
    """
    return train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE)


def save_manual_added_points(points, filename="manual_added_points.npy"):
    """
    Saves manually added points to a file.

    Args:
        points (list): List of added points.
        filename (str): Path to save the points.
    """
    np.save(filename, np.array(points, dtype=float).reshape(-1, 2))


def load_manual_added_points(filename="manual_added_points.npy"):
    """
    Loads manually added points from a NumPy file.

    Args:
        filename (str): File path to load the points.

    Returns:
        list: Loaded points.
    """
    try:
        points = np.load(filename, allow_pickle=True)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        print(f"Loaded {len(points)} points from '{filename}'")
        return points.tolist()
    except FileNotFoundError:
        print(f"No saved points found in '{filename}', starting fresh.")
        return []
