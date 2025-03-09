import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from config import PLOT_SETTINGS
from dataset import load_manual_added_points, save_manual_added_points

ax = None
fig = None
added_points = load_manual_added_points()


def plot_results_with_decision_boundary(
    classifier, X, Y, uncertain_mask, high_certainty_mask
):
    """
    Plots the dataset, decision boundary, and uncertainty region.

    Args:
        classifier (SVC): Trained SVM classifier.
        X (ndarray): Feature matrix.
        Y (ndarray): Label vector.
        uncertain_mask (ndarray): Mask for uncertain points.
        high_certainty_mask (ndarray): Mask for high-certainty points.
    """
    global fig, ax
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS["figure_size"])

    # Define grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contour(
        xx,
        yy,
        Z,
        levels=[0.5],
        colors=PLOT_SETTINGS["decision_boundary_color"],
        linewidths=PLOT_SETTINGS["decision_boundary_width"],
    )

    # Plot certain points
    ax.scatter(
        X[~uncertain_mask, 0],
        X[~uncertain_mask, 1],
        c=Y[~uncertain_mask],
        cmap="coolwarm",
        alpha=PLOT_SETTINGS["certain_point_alpha"],
        edgecolors=PLOT_SETTINGS["certain_point_edge_color"],
        label="Certain Points",
    )

    # Plot uncertain points
    ax.scatter(
        X[uncertain_mask, 0],
        X[uncertain_mask, 1],
        c=PLOT_SETTINGS["uncertain_point_color"],
        marker=PLOT_SETTINGS["uncertain_point_marker"],
        alpha=PLOT_SETTINGS["uncertain_point_alpha"],
        label="Uncertain Points",
    )

    # Convex hull for high-certainty points
    for class_label in np.unique(Y):
        class_points = X[(Y == class_label) & high_certainty_mask]
        if len(class_points) > 2:
            hull = ConvexHull(class_points)
            for simplex in hull.simplices:
                ax.plot(
                    class_points[simplex, 0],
                    class_points[simplex, 1],
                    color=PLOT_SETTINGS["hull_line_color"],
                    linewidth=PLOT_SETTINGS["hull_line_width"],
                )

    # Plot manually added points
    if added_points:
        ax.scatter(
            np.array(added_points)[:, 0],
            np.array(added_points)[:, 1],
            color=PLOT_SETTINGS["added_point_color"],
            marker=PLOT_SETTINGS["added_point_marker"],
            alpha=PLOT_SETTINGS["added_point_alpha"],
            label="Manually Added Points",
        )

    ax.set_title("SVM Classification with Uncertainty Visualization")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def onclick(event):
    """Handles mouse clicks and adds a new point."""
    global ax

    if event.xdata is None or event.ydata is None:
        return

    added_points.append([event.xdata, event.ydata])
    save_manual_added_points(added_points)

    ax.scatter(
        event.xdata,
        event.ydata,
        color=PLOT_SETTINGS["added_point_color"],
        marker=PLOT_SETTINGS["added_point_marker"],
        alpha=PLOT_SETTINGS["added_point_alpha"],
        label="Added Point",
    )
    plt.draw()
