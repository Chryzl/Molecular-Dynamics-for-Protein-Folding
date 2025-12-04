"""
Validation tests for trajectory sampling quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance


def run_validation(
    tica_coords_per_replica: list,
    trajectories: list,
    output_dir: Path,
):
    """
    Validate sampling quality across replicas.

    Args:
        tica_coords_per_replica: List of TICA coords per replica
        trajectories: List of original trajectory dicts
        output_dir: Output directory for plots

    Returns:
        dict with keys:
            - ergodicity_score: Measure of replica overlap
            - converged: Boolean indicating good overlap
    """

    print("\n[VALIDATION TESTS]")

    # Test: Ergodicity via replica overlap in TICA space
    print("Testing ergodicity (replica overlap)...")

    n_replicas = len(tica_coords_per_replica)

    # Compute pairwise Wasserstein distances in TICA space (first 2 components)
    distances = []
    for i in range(n_replicas):
        for j in range(i + 1, n_replicas):
            # Use first TICA component for 1D Wasserstein distance
            dist = wasserstein_distance(
                tica_coords_per_replica[i][:, 0],
                tica_coords_per_replica[j][:, 0],
            )
            distances.append(dist)

    mean_distance = np.mean(distances)
    print(f"  Mean Wasserstein distance (TICA-1): {mean_distance:.4f}")

    # Heuristic: Good overlap if mean distance < 0.5
    converged = mean_distance < 0.5

    if converged:
        print("  ✓ Replicas show good overlap (ergodic)")
    else:
        print("  ⚠ Replicas show poor overlap (may not be ergodic)")

    # Plot replica overlap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: TICA projections overlaid
    colors = plt.cm.tab10(np.linspace(0, 1, n_replicas))
    for i, tica_coords in enumerate(tica_coords_per_replica):
        axes[0].scatter(
            tica_coords[:, 0],
            tica_coords[:, 1],
            c=[colors[i]],
            s=1,
            alpha=0.3,
            label=f"Replica {i}",
        )

    axes[0].set_xlabel("TICA 1", fontsize=11)
    axes[0].set_ylabel("TICA 2", fontsize=11)
    axes[0].set_title("Replica Overlap (TICA Space)", fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: Loss distributions
    for i, traj in enumerate(trajectories):
        axes[1].hist(
            traj["loss"],
            bins=30,
            alpha=0.5,
            label=f"Replica {i}",
            color=colors[i],
            density=True,
        )

    axes[1].set_xlabel("Loss", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Loss Distribution Overlap", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ergodicity_check.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'ergodicity_check.png'}")

    return {
        "ergodicity_score": mean_distance,
        "converged": converged,
    }
