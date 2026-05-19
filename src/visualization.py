import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_predicted_vs_observed(observed, predicted):
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(observed, predicted, alpha=0.5, s=20) # alpha = transparency, s = size

    # draw a diagonal line
    plt.plot( 
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=2,
        label="45-degree line"
    )

    zoom_max = np.percentile(observed, 99.999) 
    plt.xlim(0, zoom_max)
    plt.ylim(0, zoom_max)

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Observed")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return

