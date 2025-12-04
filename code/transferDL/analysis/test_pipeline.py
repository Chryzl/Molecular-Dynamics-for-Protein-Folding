"""
Quick test script to verify analysis pipeline works.

This creates synthetic trajectory data to test the analysis modules.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.tica import run_tica_analysis
from analysis.msm import run_msm_analysis
from analysis.validation import run_validation
from analysis.plot_fes import plot_free_energy_surface


def create_synthetic_trajectory(n_frames=1000, n_params=100):
    """Create synthetic trajectory for testing."""
    np.random.seed(42)

    # Simulate trajectory with 2 metastable states
    state = 0
    theta_list = []
    loss_list = []

    for i in range(n_frames):
        # Random state transitions
        if np.random.rand() < 0.01:
            state = 1 - state

        # Parameters depend on state
        if state == 0:
            theta = np.random.randn(n_params) * 0.5 + 1.0
            loss = 0.3 + np.random.rand() * 0.1
        else:
            theta = np.random.randn(n_params) * 0.5 - 1.0
            loss = 0.5 + np.random.rand() * 0.1

        theta_list.append(theta)
        loss_list.append(loss)

    return {
        "theta": np.array(theta_list),
        "loss": np.array(loss_list),
        "accuracy": np.random.rand(n_frames),
    }


def main():
    print("=" * 60)
    print("TESTING ANALYSIS PIPELINE WITH SYNTHETIC DATA")
    print("=" * 60)

    # Create output directory
    output_dir = Path("./test_analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Generate synthetic trajectories
    print("\nGenerating 3 synthetic replicas...")
    trajectories = [create_synthetic_trajectory() for _ in range(3)]
    theta_per_replica = [t["theta"] for t in trajectories]
    theta_concat = np.vstack(theta_per_replica)

    print(f"  Total frames: {len(theta_concat)}, Parameters: {theta_concat.shape[1]}")

    # Run TICA
    print("\n" + "=" * 60)
    print("STEP 1: TICA")
    print("=" * 60)
    tica_results = run_tica_analysis(
        theta_concat=theta_concat,
        theta_per_replica=theta_per_replica,
        lag_time=10,
        n_components=5,
        output_dir=output_dir,
    )

    # Run MSM
    print("\n" + "=" * 60)
    print("STEP 2: MSM")
    print("=" * 60)
    msm_results = run_msm_analysis(
        tica_coords_concat=tica_results["tica_coords_concat"],
        tica_coords_per_replica=tica_results["tica_coords_per_replica"],
        n_clusters=50,
        lag_time=5,
        output_dir=output_dir,
    )

    # Run validation
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATION")
    print("=" * 60)
    validation_results = run_validation(
        tica_coords_per_replica=tica_results["tica_coords_per_replica"],
        trajectories=trajectories,
        output_dir=output_dir,
    )

    # Plot FES
    print("\n" + "=" * 60)
    print("STEP 4: FREE ENERGY SURFACE")
    print("=" * 60)
    fes_results = plot_free_energy_surface(
        tica_coords=tica_results["tica_coords_concat"],
        stationary_dist=msm_results["stationary_distribution"],
        discrete_traj=msm_results["discrete_traj_concat"],
        cluster_centers=msm_results["cluster_centers"],
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nResults in: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
