import mdtraj as md
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from deeptime.decomposition import TICA
import os

# -----------------------------
# 1. Load & align trajectories
# -----------------------------

BASE = "code/data/Chignolin/"

# file lists
xtc_files = [
    # BASE + "xtc/nvt_prod_1.xtc",
    BASE + "xtc/nvt_prod_2.xtc",
    # BASE + "xtc/nvt_prod_3.xtc",
    # BASE + "xtc/nvt_prod_4.xtc",
    # BASE + "xtc/nvt_prod_5.xtc",
]

tpr_file = BASE + "tpr/nvt_prod_1.tpr"  # can use any tpr; they should be identical
top_file = BASE + "topology.pdb"  # use matching TPR topology for the XTC files
# load reference structure for alignment
# ref = md.load(ref_pdb)

u = mda.Universe(tpr_file, xtc_files[0])

print("atoms:", len(u.atoms))  # should match the number of atoms in xtc (e.g. 12295)

# write a full PDB or GRO to use with MDTraj
if not os.path.exists(top_file):
    u.atoms.write(top_file)  # or 'full_top.gro'

trajectories = []
for xtc in xtc_files:
    traj = md.load(xtc, top=top_file)
    # traj.superpose(ref)  # align each trajectory to the crystal structure
    trajectories.append(traj)

# concatenate all trajectories
traj_all = trajectories[0].join(trajectories[1:])
for tr in trajectories[2:]:
    traj_all = traj_all.join(tr)

print("Total frames:", traj_all.n_frames)

# -----------------------------
# 2. Extract features
# -----------------------------

# Option A: use Cartesian coordinates of CA atoms
# ca_indices = traj_all.topology.select("name CA")
# xyz = traj_all.xyz[:, ca_indices, :].reshape(traj_all.n_frames, -1)

# Option B: (recommended) use pairwise CA distances
pairs = traj_all.topology.select_pairs("name CA", "name CA")
xyz = md.compute_distances(traj_all, pairs)

# -----------------------------
# 3. PCA using deeptime
# -----------------------------
tica = TICA(lagtime=10)
tica.fit(xyz)
pcs = tica.transform(xyz)  # shape = (frames, 3)

pc1, pc2, pc3 = pcs[:, 0], pcs[:, 1], pcs[:, 2]

# -----------------------------
# 4. Free energy along PCA subspace
# -----------------------------


def free_energy(x, y, kBT=2.5, bins=100):
    """Compute 2D free energy surface."""
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    F = -kBT * np.log(hist + 1e-12)  # add epsilon to avoid log(0)
    return F, xedges, yedges


F, xedges, yedges = free_energy(pc1, pc2)

# -----------------------------
# 5. Plotting
# -----------------------------

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# PCA density plot
ax[0].hist2d(pc1, pc2, bins=100, cmap="viridis")
ax[0].set_title("2D density (PC1 vs PC2)")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")

# Free energy plot
c = ax[1].imshow(
    F.T,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin="lower",
    cmap="turbo",
)
ax[1].set_title("Free Energy Surface")
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")
fig.colorbar(c, ax=ax[1])

# Histogram of PCs
ax[2].hist(pc1, bins=80, alpha=0.7, label="PC1")
ax[2].hist(pc2, bins=80, alpha=0.7, label="PC2")
ax[2].set_title("PC histograms")
ax[2].legend()

plt.tight_layout()
plt.show()

print()
