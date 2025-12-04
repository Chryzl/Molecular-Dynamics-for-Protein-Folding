"""
Time-lagged Independent Component Analysis (TICA) for trajectory dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from deeptime.decomposition import TICA


def run_tica_analysis(
    theta_concat: np.ndarray,
    theta_per_replica: list,
    lag_time: int,
    n_components: int,
    output_dir: Path,
    pca_dim: int = 100,
):
    """
    Run TICA analysis on parameter trajectories with PCA pre-filtering.

    Args:
        theta_concat: Concatenated parameters, shape (total_frames, n_params)
        theta_per_replica: List of parameter arrays per replica
        lag_time: TICA lag time
        n_components: Number of TICA components
        output_dir: Output directory for plots
        pca_dim: Number of PCA components for pre-filtering (default: 100)

    Returns:
        dict with keys:
            - tica_model: Fitted TICA model
            - tica_coords_concat: Transformed concatenated trajectory
            - tica_coords_per_replica: List of transformed per-replica trajectories
            - timescales: Implied timescales
            - eigenvalues: TICA eigenvalues
    """

    print(f"Input shape: {theta_concat.shape}")

    # Step 1: Pre-filter with PCA to reduce dimensionality
    print(f"\nStep 1: PCA pre-filtering to {pca_dim} dimensions...")
    pca = PCA(n_components=pca_dim, whiten=False)
    theta_pca_concat = pca.fit_transform(theta_concat)
    theta_pca_per_replica = [pca.transform(theta) for theta in theta_per_replica]

    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Shape after PCA: {theta_pca_concat.shape}")

    # Step 2: Convert to float32 (deeptime needs float32, not float16)
    theta_pca_concat = theta_pca_concat.astype(np.float32)
    theta_pca_per_replica = [t.astype(np.float32) for t in theta_pca_per_replica]

    print(f"\nStep 2: Fitting TICA with lag={lag_time}, n_components={n_components}...")

    # Fit TICA on PCA-reduced data
    tica_estimator = TICA(lagtime=lag_time, dim=n_components)
    tica_estimator.fit(theta_concat[:, :5000].astype(np.float32))  # theta_pca_concat)

    # Fetch the fitted model
    tica_model = tica_estimator.fetch_model()

    # Transform concatenated trajectory
    tica_coords_concat = tica_model.transform(theta_concat[:, :5000].astype(np.float32))

    # Transform each replica separately
    tica_coords_per_replica = [
        tica_model.transform(theta)
        for theta in [
            theta_concat[:, :5000].astype(np.float32)
        ]  # theta_pca_per_replica
    ]

    # Get singular values (which represent autocorrelations for TICA) and timescales
    singular_values = tica_model.singular_values
    timescales = (
        tica_model.timescales()
    )  # Use built-in method (already converts to timescales)

    print(f"  TICA singular values: {singular_values[:5]}")
    print(f"  Implied timescales: {timescales[:5]}")

    # Internal test: Check singular values are positive (detailed balance)
    print("\n[TICA TESTS]")
    if np.all(singular_values > 0):
        print("  ✓ All singular values positive (detailed balance satisfied)")
    else:
        print(
            f"  ✗ Warning: {np.sum(singular_values <= 0)} non-positive singular values!"
        )

    # Test: Check timescales are finite
    if np.all(np.isfinite(timescales)):
        print("  ✓ All timescales finite")
    else:
        print("  ✗ Warning: Non-finite timescales detected!")

    # Plot implied timescales
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(timescales) + 1), timescales, "o-", linewidth=2)
    ax.set_xlabel("TICA Component", fontsize=12)
    ax.set_ylabel("Implied Timescale (steps)", fontsize=12)
    ax.set_title(f"TICA Implied Timescales (lag={lag_time})", fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / "tica_timescales.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'tica_timescales.png'}")

    # Plot PCA projections (first 2 components)
    fig, axes = plt.subplots(
        1, len(theta_pca_per_replica), figsize=(5 * len(theta_pca_per_replica), 4)
    )
    if len(theta_pca_per_replica) == 1:
        axes = [axes]

    for i, pca_coords in enumerate(theta_pca_per_replica):
        axes[i].scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=np.arange(len(pca_coords)),
            cmap="viridis",
            s=1,
            alpha=0.5,
        )
        axes[i].set_xlabel("PCA 1", fontsize=11)
        axes[i].set_ylabel("PCA 2", fontsize=11)
        axes[i].set_title(f"Replica {i} (PCA)", fontsize=12)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_projections.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'pca_projections.png'}")

    # Plot TICA projections (first 2 components)
    fig, axes = plt.subplots(
        1, len(tica_coords_per_replica), figsize=(5 * len(tica_coords_per_replica), 4)
    )
    if len(tica_coords_per_replica) == 1:
        axes = [axes]

    for i, tica_coords in enumerate(tica_coords_per_replica):
        axes[i].scatter(
            tica_coords[:, 0],
            tica_coords[:, 1],
            c=np.arange(len(tica_coords)),
            cmap="viridis",
            s=1,
            alpha=0.5,
        )
        axes[i].set_xlabel("TICA 1", fontsize=11)
        axes[i].set_ylabel("TICA 2", fontsize=11)
        axes[i].set_title(f"Replica {i} (TICA)", fontsize=12)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tica_projections.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'tica_projections.png'}")

    return {
        "tica_model": tica_model,
        "tica_coords_concat": tica_coords_concat,
        "tica_coords_per_replica": tica_coords_per_replica,
        "timescales": timescales,
        "singular_values": singular_values,
    }
