#!/usr/bin/env python3
"""Extract representative structures for local minima in a 2D FES (PC1 vs PC2).

Usage example:
  python code/extract_minima_structures.py \
    --xtc-dir code/data/Chignolin/xtc \
    --top code/data/Chignolin/topology.pdb \
    --nminima 3 \
    --outdir structures

The script recomputes the same features and TICA transform used in
`code/Figures/Chignolin_FES.py` and finds local minima on the free-energy grid.
For each minima it picks the closest trajectory frame in (PC1,PC2) space and
saves a PDB snapshot.
"""
import argparse
import glob
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.transformations as trans
import matplotlib.pyplot as plt
from deeptime.decomposition import TICA


def free_energy(x, y, kBT=2.5, bins=100):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    F = -kBT * np.log(hist + 1e-12)
    return F, xedges, yedges


def find_local_minima(F):
    minima = []
    nx, ny = F.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            val = F[i, j]
            # compare with 8 neighbors
            neigh = F[i - 1 : i + 2, j - 1 : j + 2]
            if val <= np.min(neigh) and (neigh.shape == (3, 3)):
                # ensure strictly less than at least one neighbor to avoid flat regions
                if np.any(neigh != val):
                    minima.append((val, i, j))
    return minima


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xtc-dir", required=True)
    p.add_argument("--top", required=True)
    p.add_argument("--nminima", type=int, default=3)
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--lag", type=int, default=10)
    p.add_argument(
        "--align-selection",
        default="protein and backbone",
        help="MDAnalysis selection string used for fitting (e.g. 'name CA' or 'protein and backbone')",
    )
    p.add_argument("--outdir", default="structures")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    xtc_files = [sorted(glob.glob(os.path.join(args.xtc_dir, "*.xtc")))[1]]
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

    # compute CA Cartesian coordinates (flattened) from the aligned Universe (in memory)
    n_frames = len(u.trajectory)
    ca = u.select_atoms("name CA")
    n_ca = len(ca)
    if n_ca < 1:
        raise SystemExit("No CA atoms found to extract coordinates.")
    print(f"Total frames: {n_frames}")
    print(f"Collecting CA Cartesian coordinates (n_ca={n_ca}) per frame (in memory)...")

    # feature matrix: frames x (n_ca * 3)
    X = np.empty((n_frames, n_ca * 3), dtype=float)
    for i, ts in enumerate(u.trajectory):
        pos = ca.positions  # (n_ca, 3)
        X[i, :] = pos.reshape(-1)

    # TICA/PCA-like transform (matches the FES generation script)
    tica = TICA(lagtime=args.lag)
    tica.fit(X)
    pcs = tica.transform(X)

    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]

    # compute free energy
    F, xedges, yedges = free_energy(pc1, pc2, bins=args.bins)

    # Plot single-axis Free Energy Surface (PC1 vs PC2)
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.imshow(
            F.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="turbo",
            aspect="auto",
        )
        ax.set_title("Free Energy Surface")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(c, ax=ax, label="Free energy (kBT)")
        outfig = os.path.join(args.outdir, "fes.png")
        fig.tight_layout()
        fig.savefig(outfig, dpi=200)
        plt.close(fig)
        print(f"Saved FES plot to {outfig}")
    except Exception as e:
        print("Failed to create FES plot:", e)

    # find local minima on the grid
    minima = find_local_minima(F)
    if len(minima) == 0:
        print("No local minima found on the grid; using global minimum instead.")
        flat_idx = np.argmin(F)
        i, j = np.unravel_index(flat_idx, F.shape)
        minima = [(F[i, j], i, j)]

    # sort minima by energy
    minima_sorted = sorted(minima, key=lambda x: x[0])

    # take top n
    chosen = minima_sorted[: args.nminima]

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    results = []
    for idx, (val, i, j) in enumerate(chosen):
        xc = xcenters[i]
        yc = ycenters[j]
        # find nearest frame in PC space
        d2 = (pc1 - xc) ** 2 + (pc2 - yc) ** 2
        frame_idx = int(np.argmin(d2))
        outpath = os.path.join(args.outdir, f"minima_{idx+1}_frame{frame_idx}.pdb")
        # write snapshot from the aligned Universe (in-memory)
        u.trajectory[frame_idx]
        u.atoms.write(outpath)
        results.append((idx + 1, val, i, j, xc, yc, frame_idx, outpath))

    print("Saved representative structures:")
    for r in results:
        print(f"Minima {r[0]}: energy={r[1]:.3f}, grid=(i={r[2]},j={r[3]}), ")
        print(f"  center=(PC1={r[4]:.4f}, PC2={r[5]:.4f}), frame={r[6]}, file={r[7]}")


if __name__ == "__main__":
    main()
