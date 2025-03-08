from sklearn.metrics import classification_report
from sklearn.svm import SVC

from config import RANDOM_STATE, SVM_GAMMA, SVM_KERNEL


def train_svm_classifier(X_train, Y_train):
    """
    Trains an SVM classifier.
    """
    classifier = SVC(
        probability=True, kernel=SVM_KERNEL, gamma=SVM_GAMMA, random_state=RANDOM_STATE
    )
    classifier.fit(X_train, Y_train)
    return classifier


def evaluate_model(classifier, X_test, Y_test):
    """
    Evaluates the classifier and prints a classification report.
    """
    Y_pred = classifier.predict(X_test)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
