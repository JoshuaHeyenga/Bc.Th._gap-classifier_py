from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from config import CENTERS, CLUSTER_STD, N_SAMPLES, RANDOM_STATE


def generate_dataset():
    """
    Generates a synthetic dataset.
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
    """
    return train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE)
