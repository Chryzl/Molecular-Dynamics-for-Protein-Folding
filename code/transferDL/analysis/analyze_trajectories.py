"""
Main orchestrator for trajectory analysis using TICA and MSMs.

Usage:
    python analysis/analyze_trajectories.py --results-dir ./results --num-replicas 3
"""

import argparse
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_trajectory
from analysis.tica import run_tica_analysis
from analysis.msm import run_msm_analysis
from analysis.validation import run_validation
from analysis.plot_fes import plot_free_energy_surface


def load_all_trajectories(results_dir: str, num_replicas: int):
    """Load all replica trajectories."""
    print("\n" + "=" * 60)
    print("LOADING TRAJECTORIES")
    print("=" * 60)

    trajectories = []
    theta_all = []
    tica_coords_all = []  # Will be populated after TICA

    for i in range(num_replicas):
        traj_path = Path(results_dir) / "trajectories" / f"trajectory_replica_{i}.npz"
        if not traj_path.exists():
            print(f"Warning: Trajectory {traj_path} not found, skipping")
            continue

        traj = load_trajectory(str(traj_path))
        trajectories.append(traj)
        theta_all.append(traj["theta"])

        print(
            f"  Replica {i}: {len(traj['theta'])} frames, "
            f"{traj['theta'].shape[1]} parameters"
        )

    if len(trajectories) == 0:
        raise FileNotFoundError("No trajectory files found!")

    # Concatenate all parameter trajectories for TICA fitting
    theta_concat = np.vstack(theta_all)
    print(f"\nTotal: {len(theta_concat)} frames concatenated")

    return trajectories, theta_all, theta_concat


def main():
    parser = argparse.ArgumentParser(description="Analyze SGLD training trajectories")
    parser.add_argument(
        "--results-dir", type=str, default="./results", help="Results directory"
    )
    parser.add_argument(
        "--num-replicas", type=int, default=1, help="Number of replicas"
    )
    parser.add_argument("--lag-time", type=int, default=10, help="TICA lag time")
    parser.add_argument(
        "--n-tica-components", type=int, default=5, help="Number of TICA components"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=100, help="Number of MSM states"
    )
    parser.add_argument(
        "--msm-lag", type=int, default=10, help="MSM lag time (in saved frames)"
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=1000,
        help="PCA pre-filtering dimension (default: 1000)",
    )

    args = parser.parse_args()

    # Create analysis output directory
    output_dir = Path(args.results_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load trajectories
    trajectories, theta_all, theta_concat = load_all_trajectories(
        args.results_dir, args.num_replicas
    )

    # Step 2: TICA Analysis
    print("\n" + "=" * 60)
    print("STEP 1: TICA ANALYSIS")
    print("=" * 60)
    tica_results = run_tica_analysis(
        theta_concat=theta_concat,
        theta_per_replica=theta_all,
        lag_time=args.lag_time,
        n_components=args.n_tica_components,
        output_dir=output_dir,
        pca_dim=args.pca_dim,
    )

    # Step 3: MSM Construction
    print("\n" + "=" * 60)
    print("STEP 2: MSM CONSTRUCTION & VALIDATION")
    print("=" * 60)
    msm_results = run_msm_analysis(
        tica_coords_concat=tica_results["tica_coords_concat"],
        tica_coords_per_replica=tica_results["tica_coords_per_replica"],
        n_clusters=args.n_clusters,
        lag_time=args.msm_lag,
        output_dir=output_dir,
    )

    # Step 4: Cross-Trajectory Validation
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATION (ERGODICITY)")
    print("=" * 60)
    validation_results = run_validation(
        tica_coords_per_replica=tica_results["tica_coords_per_replica"],
        trajectories=trajectories,
        output_dir=output_dir,
    )

    # Step 5: Free Energy Surface
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

    # Step 6: Save summary
    summary = {
        "tica": {
            "lag_time": args.lag_time,
            "n_components": args.n_tica_components,
            "timescales": tica_results["timescales"].tolist(),
            "singular_values": tica_results["singular_values"].tolist(),
        },
        "msm": {
            "n_states": args.n_clusters,
            "lag_time": args.msm_lag,
            "timescales": msm_results["msm_timescales"].tolist(),
            "ck_test_passed": msm_results["ck_test_passed"],
        },
        "validation": {
            "ergodicity_score": validation_results["ergodicity_score"],
            "replicas_converged": validation_results["converged"],
        },
        "fes": {
            "min_free_energy": fes_results["min_free_energy"],
            "n_minima": fes_results["n_minima"],
        },
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print("\nKey findings:")
    print(f"  TICA slowest timescale: {tica_results['timescales'][0]:.1f} steps")
    print(f"  MSM slowest timescale: {msm_results['msm_timescales'][0]:.1f} steps")
    print(
        f"  Chapman-Kolmogorov test: {'✓ PASS' if msm_results['ck_test_passed'] else '✗ FAIL'}"
    )
    print(
        f"  Ergodicity: {'✓ Good overlap' if validation_results['converged'] else '⚠ Poor overlap'}"
    )
    print(f"  Free energy minima found: {fes_results['n_minima']}")


if __name__ == "__main__":
    main()
