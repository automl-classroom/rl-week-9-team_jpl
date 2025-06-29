"""
Analysis for DYNA-PPO Error Plots

Adopted from GitHub Copilot and https://github.com/google-research/rliable
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Analyze and visualize results from DYNA-PPO experiments using RLiable.
    """

    # Load run from hydra run
    results_dir = Path("outputs/2025-06-29/11-51-58")
    avg_returns_file = results_dir / "avg_returns.csv"
    multi_step_errors_file = results_dir / "mutli_step_errors.csv"
    one_step_errors_file = results_dir / "one_step_errors.csv"
    if (
        not avg_returns_file.exists()
        or not multi_step_errors_file.exists()
        or not one_step_errors_file.exists()
    ):
        print("Required files not found in the results directory.")
        return
    # Load average returns
    avg_returns = np.loadtxt(avg_returns_file, delimiter=",")
    steps = avg_returns[:, 0]
    avg_returns = avg_returns[:, 1:]
    # Load multi-step errors
    multi_step_errors = np.loadtxt(multi_step_errors_file, delimiter=",")
    multi_steps = multi_step_errors[:, 0]
    multi_step_errors = multi_step_errors[:, 1:]
    # Load one-step errors
    one_step_errors = np.loadtxt(one_step_errors_file, delimiter=",")
    one_steps = one_step_errors[:, 0]
    one_step_errors = one_step_errors[:, 1]

    # Create a plot for average returns and one-step errors
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    color1 = "tab:blue"
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Average Return", color=color1)
    ax1.plot(
        steps,
        avg_returns,
        marker="o",
        linestyle="-",
        color=color1,
        label="Average Returns",
    )
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("One-Step Error", color=color2)
    ax2.plot(
        one_steps,
        one_step_errors,
        marker="x",
        linestyle="--",
        color=color2,
        label="One-Step Errors",
    )

    plt.title("Average Returns and One-Step Errors Over Training Steps")
    plt.savefig("rliable_plots/avg_returns_plot.png")
    plt.close()

    # Create a plot for multi-step errors

    # Get multi-step errors at the beginning of training, middle, and end
    multi_steps = multi_steps[[0, 4, 8]]
    multi_step_errors = multi_step_errors[[0, 3, 8], :]
    k = np.arange(multi_step_errors.shape[1]) + 1
    plt.figure(figsize=(10, 6))
    plt.plot(
        k,
        multi_step_errors.T,
        marker="o",
        linestyle="-",
        label=[f"Step {int(step)}" for step in multi_steps],
    )
    plt.title("Multi-Step Errors Over Imagination Steps")
    plt.xlabel("Imagination Steps")
    plt.ylabel("Multi-Step Error")
    plt.grid(True)
    plt.legend()
    plt.savefig("rliable_plots/multi_step_errors_plot.png")


if __name__ == "__main__":
    main()
