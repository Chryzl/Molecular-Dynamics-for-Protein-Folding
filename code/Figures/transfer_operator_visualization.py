#!/usr/bin/env python3
"""
Transfer Operator Formalism Visualization
==========================================
Generates a visualization of the transfer operator formalism showing:
(a) Potential energy and stationary distribution
(b) Dominant eigenfunctions (psi)
(c) Weighted eigenfunctions (phi)

Output: PDF figure saved to figures/transfer_operator_tau20.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.interpolate import UnivariateSpline
import os

# ==========================================
# 1. Define the Potential Energy Surface U(x)
# ==========================================
x_grid = np.linspace(1, 100, 101)
kt = 1.0  # Thermal energy

# Key points [x, energy] to match the visual shape of Prinz et al.
# Wells at approx: 12, 38, 62, 88
# Barriers at: 25, 50, 75
key_points_x = [1, 12, 25, 38, 50, 62, 75, 88, 100]
key_points_y = [10, 2, 6, 3, 9, 1.5, 4, 2, 10]

# Fit a spline
spline = UnivariateSpline(key_points_x, key_points_y, s=0, k=3)
U = spline(x_grid)

# Calculate Stationary Density (Boltzmann)
mu = np.exp(-U / kt)
mu /= np.sum(mu)  # Normalize

# ==========================================
# 2. Construct the Transition Matrix (Dynamics)
# ==========================================
# We use a rate matrix K for diffusion in 1D potential
# Rate k_ij = D * exp(-(U_j - U_i) / 2kT) for neighbors
# This ensures Detailed Balance.

N = len(x_grid)
K = np.zeros((N, N))
D = 1.0  # Diffusion coefficient

for i in range(N):
    for j in [i - 1, i + 1]:  # Neighbors
        if 0 <= j < N:
            # Arrhenius-like rate
            rate = D * np.exp(-(U[j] - U[i]) / (2 * kt))
            K[i, j] = rate

# Fill diagonals (row sums must be 0 for rate matrix)
for i in range(N):
    K[i, i] = -np.sum(K[i, :])

# Compute Transition Matrix P(tau) = exp(K * tau)
tau = 20.0  # Lag time (adjust to change "fuzziness" of transitions)
P = expm(K * tau)

# ==========================================
# 3. Spectral Analysis (Eigen Decomposition)
# ==========================================
# P is row-stochastic.
# Right eigenvectors (psi) = dynamical processes (observables)
# Left eigenvectors (phi) = densities

eigvals, eigvecs_R = np.linalg.eig(P)

# Sort by eigenvalue magnitude (descending)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
psi = eigvecs_R[:, idx]

# Ensure real parts (numerical noise can cause complex numbers)
eigvals = np.real(eigvals)
psi = np.real(psi)

# The Left eigenvectors (phi) are essentially psi weighted by mu
# (For reversible dynamics: phi_i = psi_i * mu)
phi = np.zeros_like(psi)
for i in range(len(eigvals)):
    phi[:, i] = psi[:, i] * mu

# Normalize vectors for plotting
for i in range(4):
    psi[:, i] /= np.max(np.abs(psi[:, i]))
    phi[:, i] /= np.max(np.abs(phi[:, i]))

# ==========================================
# 4. Plotting (Recreating the Figure)
# ==========================================
# IEEE column width is ~3.5 inches.
fig, axes = plt.subplots(3, 1, figsize=(3.5, 6), constrained_layout=True)

# Font sizes for IEEE style
plt.rcParams.update({"font.size": 8})
label_fontsize = 8
title_fontsize = 9

# (a) Potential & Probability
ax = axes[0]
ax.plot(x_grid, U, "k-", linewidth=1.5, label="Energy")
ax.set_ylabel("Energy (solid)", fontsize=label_fontsize)
ax.set_ylim(-1, 12)
ax.grid(axis="x", linestyle=":", alpha=0.6)
ax.tick_params(axis="both", which="major", labelsize=label_fontsize)

# Overlay probability on twin axis
ax2 = ax.twinx()
ax2.plot(x_grid, mu, "k--", linewidth=1.0, label="Probability $\mu(x)$")
ax2.set_ylabel("Probability $\mu(x)$ (dashed)", fontsize=label_fontsize)
ax2.set_yticks([])  # Hide ticks
ax2.fill_between(x_grid, mu, color="gray", alpha=0.3)

# Title
ax.set_title("(a)", loc="left", fontsize=title_fontsize, pad=10)

# (b) Dominant Eigenfunctions (psi)
ax = axes[1]
spacing = 2.5
offsets = [spacing * i for i in range(3, -1, -1)]  # [7.5, 5.0, 2.5, 0.0]
colors = ["#C0392B", "#E67E22", "#7DCEA0", "#2980B9"]  # Red, Orange, Green, Blue

# Manual sign flip to match reference (eigenvectors are arbitrary up to sign)
signs = [1, -1, -1, -1]

for i in range(4):
    y_data = psi[:, i] * signs[i]
    ax.plot(x_grid, y_data + offsets[i], color=colors[i], linewidth=1.5)
    ax.axhline(offsets[i], color="k", linewidth=0.5, linestyle="--")
    ax.fill_between(x_grid, y_data + offsets[i], offsets[i], color=colors[i], alpha=0.3)

ax.set_yticks(offsets)
ax.set_yticklabels(
    [r"$\psi_1$", r"$\psi_2$", r"$\psi_3$", r"$\psi_4$"], fontsize=label_fontsize
)
ax.grid(axis="x", linestyle=":", alpha=0.6)
ax.tick_params(axis="x", labelsize=label_fontsize)
ax.set_title("(b)", loc="left", fontsize=title_fontsize, pad=10)

# (c) Weighted Eigenfunctions (phi)
ax = axes[2]
for i in range(4):
    y_data = phi[:, i] * signs[i]
    ax.plot(x_grid, y_data + offsets[i], color=colors[i], linewidth=1.5)
    ax.axhline(offsets[i], color="k", linewidth=0.5, linestyle="--")
    ax.fill_between(x_grid, y_data + offsets[i], offsets[i], color=colors[i], alpha=0.3)

ax.set_yticks(offsets)
ax.set_yticklabels(
    [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"], fontsize=label_fontsize
)
ax.grid(axis="x", linestyle=":", alpha=0.6)
ax.tick_params(axis="x", labelsize=label_fontsize)
ax.set_title("(c)", loc="left", fontsize=title_fontsize, pad=10)

# ==========================================
# 5. Save as PDF
# ==========================================
output_path = os.path.join("figures", f"transfer_operator_tau{int(tau)}.pdf")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
print(f"Figure saved to: {output_path}")

plt.close()
