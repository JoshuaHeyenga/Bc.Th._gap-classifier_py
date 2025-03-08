import numpy as np

from config import UNCERTAINTY_THRESHOLD
from dataset import generate_dataset, split_dataset
from model import evaluate_model, train_svm_classifier
from visualization import plot_results_with_decision_boundary


def compute_uncertainty_regions(classifier, X, Y, threshold=UNCERTAINTY_THRESHOLD):
    """
    Identifies uncertainty and high-certainty regions.
    """
    probabilities = classifier.predict_proba(X)[:, 1]
    uncertain_mask = (probabilities > threshold) & (probabilities < (1 - threshold))
    high_certainty_mask = (probabilities <= 0.01) | (probabilities >= 0.99)

    Y_extended = np.copy(Y)
    Y_extended[uncertain_mask] = 2  # Assign a new class for uncertain points
    return uncertain_mask, high_certainty_mask, Y_extended


def main():
    """
    Main function to execute the workflow.
    """
    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    classifier = train_svm_classifier(X_train, Y_train)
    evaluate_model(classifier, X_test, Y_test)

    uncertain_mask, high_certainty_mask, Y_extended = compute_uncertainty_regions(
        classifier, X, Y
    )
    plot_results_with_decision_boundary(
        classifier, X, Y_extended, uncertain_mask, high_certainty_mask
    )


if __name__ == "__main__":
    main()
