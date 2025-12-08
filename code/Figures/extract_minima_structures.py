#!/usr/bin/env python3
"""Plot Free Energy Surface (FES) using TICA and MSM weights.

Usage example:
  python code/Figures/extract_minima_structures.py \
    --xtc-dir code/data/Chignolin/xtc \
    --top code/data/Chignolin/topology.pdb \
    --outdir figures

The script computes TICA features (pairwise CA distances), builds an MSM,
and plots the reweighted Free Energy Surface.
"""
import argparse
import glob
import os
import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as trans
import matplotlib.pyplot as plt
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.util import energy2d
from tqdm import tqdm

# Set matplotlib style for publication quality
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "font.family": "serif",
    }
)


def compute_ca_pairwise_distances(universe, ca_atoms):
    """Compute pairwise distances between CA atoms.

    Args:
        universe: MDAnalysis Universe
        ca_atoms: AtomGroup of CA atoms

    Returns:
        Array of shape (n_frames, n_pairs) with pairwise distances
    """
    n_ca = len(ca_atoms)
    n_pairs = (n_ca * (n_ca - 1)) // 2
    n_frames = len(universe.trajectory)

    distances = np.empty((n_frames, n_pairs), dtype=float)

    # Create pairs indices
    pairs = []
    for i in range(n_ca):
        for j in range(i + 1, n_ca):
            pairs.append((i, j))

    for frame_idx, ts in enumerate(
        tqdm(universe.trajectory, desc="Computing CA distances")
    ):
        pos = ca_atoms.positions
        for pair_idx, (i, j) in enumerate(pairs):
            distances[frame_idx, pair_idx] = np.linalg.norm(pos[i] - pos[j])

    return distances


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xtc-dir", default="code/data/Chignolin/xtc")
    p.add_argument("--top", default="code/data/Chignolin/topology.pdb")
    p.add_argument(
        "--n-clusters", type=int, default=100, help="Number of clusters for KMeans"
    )
    p.add_argument(
        "--bins", type=int, default=80, help="Number of bins for energy2d grid"
    )
    p.add_argument("--lag", type=int, default=10, help="Lagtime for TICA and MSM")
    p.add_argument("--kbt", type=float, default=2.5, help="kBT for energy calculation")
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    xtc_files = sorted(glob.glob(os.path.join(args.xtc_dir, "*.xtc")))
    if len(xtc_files) == 0:
        raise SystemExit(f"No .xtc files found in {args.xtc_dir}")

    # Load all trajectories into a single Universe and apply on-the-fly transforms
    print("Loading Universe with trajectories (will keep aligned frames in memory)...")
    u = mda.Universe(args.top, *xtc_files)
    protein = u.select_atoms("protein")
    not_protein = u.select_atoms("not protein")

    transforms = [
        trans.unwrap(protein),
        trans.center_in_box(protein, wrap=True),
        trans.wrap(not_protein),
    ]
    u.trajectory.add_transformations(*transforms)

    # Select CA atoms
    ca = u.select_atoms("name CA")
    n_ca = len(ca)
    if n_ca < 1:
        raise SystemExit("No CA atoms found to extract coordinates.")
    n_frames = len(u.trajectory)
    print(f"Total frames: {n_frames}")
    print(f"Number of CA atoms: {n_ca}")

    # Compute pairwise CA distances as features
    print("Computing pairwise CA distances...")
    X = compute_ca_pairwise_distances(u, ca)
    print(f"Feature matrix shape: {X.shape}")

    # TICA transform
    print(f"Fitting TICA with lagtime={args.lag}...")
    tica = TICA(lagtime=args.lag)
    tica.fit(X)
    pcs = tica.transform(X)
    print(f"TICA output shape: {pcs.shape}")

    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]
    traj_concat = np.column_stack([pc1, pc2])

    # KMeans clustering in TICA space
    print(f"Clustering with {args.n_clusters} clusters...")
    clustering = KMeans(n_clusters=args.n_clusters).fit_fetch(traj_concat)
    dtraj = clustering.transform(traj_concat)

    # Build MSM
    print(f"Building MSM with lagtime={args.lag}...")
    msm = MaximumLikelihoodMSM(lagtime=args.lag).fit_fetch([dtraj])

    # Compute trajectory weights
    print("Computing trajectory weights from MSM...")
    weights = msm.compute_trajectory_weights([dtraj])[0]

    # Compute free energy surface using energy2d with MSM weights
    print("Computing free energy surface...")
    energies = energy2d(
        pc1,
        pc2,
        bins=(args.bins, args.bins),
        kbt=args.kbt,
        weights=weights,
        shift_energy=True,
    )

    # Plot Free Energy Surface using energy2d plot method
    print("Creating FES plot...")
    # IEEE column width is ~3.5 inches. We make it slightly thinner.
    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # Plot FES
    ax, contour, cbar = energies.plot(ax=ax, contourf_kws=dict(cmap="nipy_spectral"))

    cbar.set_label("Free Energy / $k_B T$")
    ax.set_title(f"FES (TICA lag={args.lag})")
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")

    outfig = os.path.join(args.outdir, f"fes_msm_tau{args.lag}.pdf")
    fig.tight_layout()
    fig.savefig(outfig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MSM-weighted FES plot to {outfig}")

    print("\nDone! Free energy surface computed using MSM-weighted approach.")


if __name__ == "__main__":
    main()
