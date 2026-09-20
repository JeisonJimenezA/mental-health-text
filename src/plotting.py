"""Confusion-matrix plotting shared across notebooks."""
import numpy as np

BAND_LABELS = ["Minimal", "Mild", "Moderate", "Mod-Severe/Severe"]
SCREENING_LABELS = ["Negative", "Positive"]


def plot_confusion_matrix(ax, matrix, labels, title):
    matrix = np.array(matrix)
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    threshold = matrix.max() / 2 if matrix.max() > 0 else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > threshold else "black")
