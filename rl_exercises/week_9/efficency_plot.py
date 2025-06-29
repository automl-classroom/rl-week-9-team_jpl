"""
RLiable Analysis for DYNA-PPO Sample Efficiency

Adopted from GitHub Copilot and https://github.com/google-research/rliable
"""

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils


def main():
    """
    Analyze and visualize results from DYNA-PPO experiments using RLiable.
    """

    # Load runs
    results_dir = Path("Log_returns_eval")
    baseline_data = {}
    training_steps = np.array([])
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        all_eval_rewards = []

        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name

            with open(seed_dir / "metrics.csv", "r") as f:
                lines = f.readlines()[1:]  # Skip header line
                if len(lines) < 2:
                    print(f"No metrics found for {model_name} seed {seed}. Skipping.")
                    continue

                # Parse CSV lines
                reader = csv.DictReader(io.StringIO("step,return\n" + "".join(lines)))
                rows = list(reader)
                returns_list = [float(row["return"]) for row in rows]
                eval_returns = np.array(returns_list)
                all_eval_rewards.append(eval_returns)
                if training_steps.size == 0:
                    training_steps = np.array([int(row["step"]) for row in rows])

        if all_eval_rewards:
            eval_rewards = np.array(all_eval_rewards)
            # Reshape to (n_seeds, 1, n_evals) for RLiable compatibility
            eval_rewards = eval_rewards.reshape(
                eval_rewards.shape[0], 1, eval_rewards.shape[1]
            )
            baseline_data[model_name] = eval_rewards

    if not baseline_data:
        print("No valid data found in results directory.")
        return

    # Joint Plot
    ##############
    # Use every second evaluation for "ppo"
    if "ppo" in baseline_data:
        baseline_data["ppo"] = baseline_data["ppo"][:, :, ::2]
    if "dyna_ppo" in baseline_data:
        baseline_data["dyna_ppo"] = baseline_data["dyna_ppo"][:, :, ::2]
    if "dyna_ppo_HPO" in baseline_data:
        baseline_data["dyna_ppo_HPO"] = baseline_data["dyna_ppo_HPO"][
            :, :, : baseline_data["dyna_ppo"].shape[2]
        ]
    if "ppo_HPO" in baseline_data:
        baseline_data["ppo_HPO"] = baseline_data["ppo_HPO"][
            :, :, : baseline_data["ppo"].shape[2]
        ]
    training_steps = training_steps[::2]

    ### Plot Runs over Time ###

    def iqm(scores):
        """
        Compute the Inverse of the Interquartile Mean (IQM) for given scores.
        """
        return np.array(
            [
                metrics.aggregate_mean(scores[..., frame])
                for frame in range(scores.shape[-1])
            ]
        )

    iqm_scores, iqm_cis = rly.get_interval_estimates(baseline_data, iqm, reps=10000)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use RLiable's plot function
    plot_utils.plot_sample_efficiency_curve(
        training_steps / 1000,  # Convert to thousands
        iqm_scores,
        iqm_cis,
        algorithms=list(baseline_data.keys()),
        xlabel="Training Steps (thousands)",
        ylabel="Mean Evaluation Return",
        ax=ax,
    )

    # Add title and legend
    ax.set_title("PPO and Dyna-PPO Comparison with 10 Seeds per Model after HPO")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = Path("rliable_plots")
    plt.savefig(
        save_path / "HPO2_ppo_dyna_ppo_comparison_sample_efficiency_rliable.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    main()
