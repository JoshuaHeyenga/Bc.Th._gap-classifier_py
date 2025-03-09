# Dataset settings
N_SAMPLES = 500
CENTERS = 2
CLUSTER_STD = 2.0
RANDOM_STATE = 42

# Model settings
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

# Uncertainty threshold
UNCERTAINTY_THRESHOLD = 0.05

# Visualization settings
PLOT_SETTINGS = {
    "decision_boundary_color": "grey",
    "decision_boundary_width": 1.5,
    "certain_point_alpha": 0.5,
    "certain_point_edge_color": "black",
    "uncertain_point_color": "grey",
    "uncertain_point_marker": "D",
    "uncertain_point_alpha": 0.7,
    "added_point_color": "orange",
    "added_point_marker": "D",
    "added_point_alpha": 0.7,
    "hull_line_color": "black",
    "hull_line_width": 1.5,
    "figure_size": (8, 6),
}
