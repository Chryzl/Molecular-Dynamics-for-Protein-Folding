"""
Free Energy Surface plotting from MSM stationary distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


def plot_free_energy_surface(
    tica_coords: np.ndarray,
    stationary_dist: np.ndarray,
    discrete_traj: np.ndarray,
    cluster_centers: np.ndarray,
    output_dir: Path,
    temperature: float = 1.0,
):
    """
    Generate free energy surface from MSM stationary distribution.

    Args:
        tica_coords: TICA coordinates (n_frames, n_tica)
        stationary_dist: MSM stationary distribution (n_states,)
        discrete_traj: Discretized trajectory mapping frames to states
        cluster_centers: Cluster centers in TICA space (n_states, n_tica)
        output_dir: Output directory
        temperature: k_B * T (default 1.0 for dimensionless)

    Returns:
        dict with keys:
            - min_free_energy: Minimum free energy value
            - n_minima: Number of local minima detected
    """

    print("Generating free energy surface...")

    # Use first 2 TICA components for 2D FES
    tica1 = tica_coords[:, 0]
    tica2 = tica_coords[:, 1]

    # Create 2D histogram grid
    n_bins = 50
    H, xedges, yedges = np.histogram2d(tica1, tica2, bins=n_bins, density=True)

    # For each grid cell, compute average stationary probability
    # by mapping histogram bins to states
    grid_prob = np.zeros_like(H)

    for i in range(len(tica1)):
        # Find which bin this point belongs to
        x_idx = np.searchsorted(xedges[:-1], tica1[i], side="right") - 1
        y_idx = np.searchsorted(yedges[:-1], tica2[i], side="right") - 1

        x_idx = np.clip(x_idx, 0, n_bins - 1)
        y_idx = np.clip(y_idx, 0, n_bins - 1)

        # Add stationary probability of this state
        state = discrete_traj[i]
        if state < len(stationary_dist):
            grid_prob[x_idx, y_idx] += stationary_dist[state]

    # Normalize grid probabilities
    grid_prob /= grid_prob.sum() + 1e-10

    # Compute free energy: F = -k_B * T * ln(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        free_energy = -temperature * np.log(grid_prob + 1e-10)

    # Set infinite values to max finite value
    max_finite = np.max(free_energy[np.isfinite(free_energy)])
    free_energy[~np.isfinite(free_energy)] = max_finite

    # Smooth slightly for visualization
    free_energy_smooth = gaussian_filter(free_energy, sigma=1.0)

    # Shift so minimum is at 0
    free_energy_smooth -= free_energy_smooth.min()

    min_free_energy = free_energy_smooth.min()

    # Detect local minima
    # Use negative to find peaks in inverted surface
    peaks_x, peaks_y = find_peaks(-free_energy_smooth.ravel(), height=None)[0], []
    n_minima = len(peaks_x)

    print(f"  Free energy range: 0 to {free_energy_smooth.max():.2f}")
    print(f"  Local minima detected: {n_minima}")

    # Plot free energy surface
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use center of bins for plotting
    X = (xedges[:-1] + xedges[1:]) / 2
    Y = (yedges[:-1] + yedges[1:]) / 2

    # Create contour plot
    levels = np.linspace(0, np.percentile(free_energy_smooth, 95), 20)
    contourf = ax.contourf(X, Y, free_energy_smooth.T, levels=levels, cmap="viridis")
    contour = ax.contour(
        X,
        Y,
        free_energy_smooth.T,
        levels=levels,
        colors="white",
        linewidths=0.5,
        alpha=0.3,
    )

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label(r"Free Energy ($k_B T$)", fontsize=12)

    # Overlay trajectory as scatter
    ax.scatter(tica1, tica2, c="white", s=0.1, alpha=0.05)

    ax.set_xlabel("TICA 1", fontsize=12)
    ax.set_ylabel("TICA 2", fontsize=12)
    ax.set_title("Free Energy Surface (Loss Landscape)", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "free_energy_surface.png", dpi=200)
    plt.close()
    print(f"  Saved: {output_dir / 'free_energy_surface.png'}")

    return {
        "min_free_energy": float(min_free_energy),
        "n_minima": n_minima,
    }
