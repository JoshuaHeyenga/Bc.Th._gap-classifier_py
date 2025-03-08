import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def plot_results_with_decision_boundary(
    classifier, X, Y, uncertain_mask, high_certainty_mask
):
    """
    Plots the dataset, decision boundary, and uncertainty region.
    """
    plt.figure(figsize=(8, 6))

    # Define grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors="grey", linewidths=1.5)

    # Uncertainty region
    probas = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    uncertain_region_mask = (probas > 0.05) & (probas < 0.95)
    uncertain_region = uncertain_region_mask.reshape(xx.shape)
    plt.contourf(xx, yy, uncertain_region, levels=[0, 1], colors="purple", alpha=0.2)

    # Plot certain points
    plt.scatter(
        X[~uncertain_mask, 0],
        X[~uncertain_mask, 1],
        c=Y[~uncertain_mask],
        cmap="coolwarm",
        alpha=0.5,
        edgecolors="black",
        label="Certain Points",
    )

    # Plot uncertain points
    plt.scatter(
        X[uncertain_mask, 0],
        X[uncertain_mask, 1],
        c="purple",
        marker="s",
        alpha=0.7,
        label="Uncertain Points",
    )

    # Convex hull for high-certainty points
    for class_label in np.unique(Y):
        class_points = X[(Y == class_label) & high_certainty_mask]
        if len(class_points) > 2:
            hull = ConvexHull(class_points)
            for simplex in hull.simplices:
                plt.plot(
                    class_points[simplex, 0],
                    class_points[simplex, 1],
                    "k-",
                    linewidth=1.5,
                )

    plt.title("SVM Classification with Uncertainty Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
