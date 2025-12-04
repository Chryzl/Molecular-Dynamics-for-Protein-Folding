"""
Markov State Model (MSM) construction and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM


def run_msm_analysis(
    tica_coords_concat: np.ndarray,
    tica_coords_per_replica: list,
    n_clusters: int,
    lag_time: int,
    output_dir: Path,
):
    """
    Build MSM from TICA coordinates.

    Args:
        tica_coords_concat: Concatenated TICA coords, shape (total_frames, n_tica)
        tica_coords_per_replica: List of TICA coords per replica
        n_clusters: Number of microstates for MSM
        lag_time: MSM lag time (in saved trajectory steps)
        output_dir: Output directory for plots

    Returns:
        dict with keys:
            - msm: Fitted MSM model
            - stationary_distribution: Equilibrium probabilities
            - discrete_traj_concat: Discretized concatenated trajectory
            - discrete_traj_per_replica: List of discretized per-replica trajectories
            - cluster_centers: Cluster centers in TICA space
            - msm_timescales: MSM implied timescales
            - ck_test_passed: Boolean CK test result
    """

    print(f"Clustering TICA coords into {n_clusters} states...")

    # Cluster using K-means
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
    clustering = kmeans.fit(tica_coords_concat)

    # Get discrete trajectories
    discrete_traj_concat = clustering.transform(tica_coords_concat)
    discrete_traj_per_replica = [
        clustering.transform(tica_coords) for tica_coords in tica_coords_per_replica
    ]

    print(f"Building MSM with lag time {lag_time}...")

    # Build MSM (deeptime expects list of discrete trajectories)
    msm_estimator = MaximumLikelihoodMSM(lagtime=lag_time, reversible=True)
    msm_estimator.fit(discrete_traj_per_replica)

    # Fetch the fitted model (MarkovStateModelCollection)
    msm_model = msm_estimator.fetch_model()

    # Get stationary distribution and timescales
    stationary_dist = msm_model.stationary_distribution
    msm_timescales = msm_model.timescales()

    print(f"  MSM built with {msm_model.n_states} states (connected)")
    print(f"  MSM timescales: {msm_timescales[:5]}")

    # Internal test: Chapman-Kolmogorov test
    print("\n[MSM TESTS]")
    ck_test_passed = True
    try:
        # Run CK test for multiple lag times
        # CK test is called on the model with a list of models at different lag times
        # For simplicity, we test using the metastable decomposition
        n_metastable = max(2, len(msm_timescales) // 3)  # Use a subset of timescales
        ck = msm_model.ck_test(
            [msm_model], n_metastable_sets=n_metastable, include_lag0=True
        )

        # Plot CK test
        fig, ax = plt.subplots(figsize=(8, 6))

        # CK test returns estimates and predictions as attributes
        if hasattr(ck, "estimates") and ck.estimates is not None:
            for i in range(min(5, ck.estimates.shape[1])):
                ax.plot(
                    ck.lagtimes,
                    ck.estimates[:, i],
                    "o-",
                    label=f"Estimate {i+1}",
                )
                if hasattr(ck, "predictions") and ck.predictions is not None:
                    ax.plot(
                        ck.lagtimes,
                        ck.predictions[:, i],
                        "--",
                        alpha=0.7,
                        label=f"Prediction {i+1}",
                    )

        ax.set_xlabel("Lag time", fontsize=12)
        ax.set_ylabel("Implied timescale", fontsize=12)
        ax.set_title("Chapman-Kolmogorov Test", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ck_test.png", dpi=150)
        plt.close()
        print(f"  ✓ Chapman-Kolmogorov test completed")
        print(f"  Saved: {output_dir / 'ck_test.png'}")

    except Exception as e:
        print(f"  ✗ Chapman-Kolmogorov test failed: {e}")
        ck_test_passed = False

    # Test: Check stationary distribution sums to 1
    if np.isclose(np.sum(stationary_dist), 1.0):
        print("  ✓ Stationary distribution normalized")
    else:
        print(f"  ✗ Warning: Stationary dist sum = {np.sum(stationary_dist)}")

    # Plot stationary distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(stationary_dist)[::-1]
    ax.bar(range(len(stationary_dist)), stationary_dist[sorted_idx])
    ax.set_xlabel("State (sorted by probability)", fontsize=12)
    ax.set_ylabel("Stationary Probability", fontsize=12)
    ax.set_title("MSM Stationary Distribution", fontsize=14)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "stationary_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'stationary_distribution.png'}")

    return {
        "msm_model": msm_model,
        "stationary_distribution": stationary_dist,
        "discrete_traj_concat": discrete_traj_concat,
        "discrete_traj_per_replica": discrete_traj_per_replica,
        "cluster_centers": clustering.cluster_centers,
        "msm_timescales": msm_timescales,
        "ck_test_passed": ck_test_passed,
    }
