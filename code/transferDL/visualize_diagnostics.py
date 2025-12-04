"""Visualize SGLD diagnostics from training runs."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse


def plot_loss_trajectory(metrics, phase="phase3", save_path=None):
    """Plot loss trajectory with rolling statistics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    steps = np.array(metrics[phase]["step"])
    losses = np.array(metrics[phase]["loss"])

    # Plot raw trajectory
    ax = axes[0]
    ax.plot(steps, losses, "b-", alpha=0.5, linewidth=1, label="Loss")

    # Add rolling mean and std
    window = 100
    if len(losses) > window:
        rolling_mean = np.convolve(losses, np.ones(window) / window, mode="valid")
        rolling_std = np.array(
            [np.std(losses[max(0, i - window) : i]) for i in range(window, len(losses))]
        )
        steps_rolling = steps[window - 1 : -1]

        ax.plot(
            steps_rolling,
            rolling_mean,
            "r-",
            linewidth=2,
            label=f"Rolling mean ({window})",
        )
        ax.fill_between(
            steps_rolling,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.3,
            color="red",
            label="±1 std",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{phase.capitalize()}: Loss Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot histogram of losses (check for stationarity)
    ax = axes[1]

    # Split into quarters to check evolution
    n = len(losses)
    quarters = [
        losses[: n // 4],
        losses[n // 4 : n // 2],
        losses[n // 2 : 3 * n // 4],
        losses[3 * n // 4 :],
    ]

    for i, quarter in enumerate(quarters):
        ax.hist(quarter, bins=30, alpha=0.5, label=f"Quarter {i+1}", density=True)

    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title("Loss Distribution Evolution (should overlap if stationary)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_autocorrelation(metrics, phase="phase3", max_lag=100, save_path=None):
    """Plot autocorrelation function of loss."""
    losses = np.array(metrics[phase]["loss"])

    # Compute autocorrelation
    losses_centered = losses - np.mean(losses)
    autocorr = np.correlate(losses_centered, losses_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr = autocorr / autocorr[0]  # Normalize
    autocorr = autocorr[:max_lag]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    lags = np.arange(len(autocorr))
    ax.bar(lags, autocorr, width=1.0, alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axhline(
        y=0.1,
        color="r",
        linestyle="--",
        linewidth=1,
        label="10% threshold (low correlation)",
    )

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{phase.capitalize()}: Loss Autocorrelation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text with decay info
    lag_below_threshold = np.where(autocorr < 0.1)[0]
    if len(lag_below_threshold) > 0:
        decay_lag = lag_below_threshold[0]
        ax.text(
            0.7,
            0.9,
            f"Decorrelation lag: {decay_lag}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_diagnostic_summary(metrics_path):
    """Print summary statistics for diagnostics."""
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    for phase in ["phase2", "phase3"]:
        if phase not in metrics or not metrics[phase]["loss"]:
            continue

        losses = np.array(metrics[phase]["loss"])

        print(f"\n{phase.upper()}:")
        print(f"  Samples: {len(losses)}")
        print(f"  Mean loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"  Min/Max: {np.min(losses):.4f} / {np.max(losses):.4f}")

        # Check drift
        mid = len(losses) // 2
        first_half_mean = np.mean(losses[:mid])
        second_half_mean = np.mean(losses[mid:])
        drift_pct = abs(second_half_mean - first_half_mean) / first_half_mean * 100

        print(f"  Drift (1st vs 2nd half): {drift_pct:.2f}%", end="")
        if drift_pct > 5:
            print(" ⚠️  (HIGH - may not be stationary)")
        else:
            print(" ✓")

        # Compute ESS
        losses_centered = losses - np.mean(losses)
        autocorr = np.correlate(losses_centered, losses_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / autocorr[0]

        sum_autocorr = 0
        for lag in range(1, min(100, len(autocorr))):
            if autocorr[lag] < 0:
                break
            sum_autocorr += autocorr[lag]

        ess = len(losses) / (1 + 2 * sum_autocorr)
        ess_ratio = ess / len(losses)

        print(f"  ESS: {ess:.0f} / {len(losses)} = {ess_ratio*100:.1f}%", end="")
        if ess_ratio < 0.1:
            print(" ⚠️  (LOW - highly correlated)")
        else:
            print(" ✓")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize SGLD diagnostics")
    parser.add_argument(
        "--replica", type=int, default=0, help="Replica ID to visualize (default: 0)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Results directory (default: ./results)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="phase3",
        choices=["phase2", "phase3"],
        help="Which phase to visualize (default: phase3)",
    )

    args = parser.parse_args()

    # Load metrics
    metrics_path = (
        Path(args.results_dir) / "metrics" / f"metrics_replica_{args.replica}.json"
    )

    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Have you run training yet? (python main.py)")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Print summary
    print_diagnostic_summary(metrics_path)

    # Create plots
    output_dir = Path(args.results_dir) / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating diagnostic plots for {args.phase}...")

    plot_loss_trajectory(
        metrics,
        phase=args.phase,
        save_path=output_dir / f"replica_{args.replica}_{args.phase}_trajectory.png",
    )

    plot_autocorrelation(
        metrics,
        phase=args.phase,
        save_path=output_dir / f"replica_{args.replica}_{args.phase}_autocorr.png",
    )

    print(f"\nDiagnostic plots saved to {output_dir}/")
    print("\nInterpretation:")
    print("  1. Loss trajectory should fluctuate around constant mean (stationary)")
    print("  2. Loss distributions should overlap across quarters")
    print("  3. Autocorrelation should decay quickly to near-zero")
    print("  4. ESS ratio > 10% indicates reasonable independence")


if __name__ == "__main__":
    main()
