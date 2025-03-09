from sklearn.metrics import classification_report
from sklearn.svm import SVC

from config import RANDOM_STATE, SVM_GAMMA, SVM_KERNEL


def train_svm_classifier(X_train, Y_train):
    """
    Trains an SVM classifier with probability estimation.

    Args:
        X_train (ndarray): Training feature matrix.
        Y_train (ndarray): Training labels.

    Returns:
        classifier (SVC): Trained SVM model.
    """
    classifier = SVC(
        probability=True, kernel=SVM_KERNEL, gamma=SVM_GAMMA, random_state=RANDOM_STATE
    )
    classifier.fit(X_train, Y_train)
    return classifier


def evaluate_model(classifier, X_test, Y_test):
    """
    Evaluates the trained model and prints a classification report.

    Args:
        classifier (SVC): Trained SVM model.
        X_test (ndarray): Test feature matrix.
        Y_test (ndarray): Test labels.
    """
    Y_pred = classifier.predict(X_test)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
