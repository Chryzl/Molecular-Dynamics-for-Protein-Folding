import numpy as np
import matplotlib.pyplot as plt

# Set style parameters for IEEE
plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "serif",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "text.usetex": False,  # Use standard text rendering for simplicity/speed
    }
)

# IEEE Column width is approx 3.5 inches
width = 3.5
height = width - 0.5  # Half an inch shorter than wide

fig, ax = plt.subplots(figsize=(width, height))

# Define the reaction coordinate
x = np.linspace(-2.0, 2.0, 1000)

# Define the potential (Free Energy)
# Steeper base funnel: 1.2 * x^2 (vs shallow quadratic)
# Ruggedness: High frequency sine wave (0.3 * sin(15*x))
# Deep native state: Gaussian well (-2.0 * exp(-10*x^2))
G = 1.2 * x**2 + 0.3 * np.sin(15 * x) - 2.0 * np.exp(-10 * x**2)

# Shift min to 0 for nicer plotting
G = G - np.min(G)

# Plot the landscape
ax.plot(x, G, color="blue", linewidth=1.5)

# Remove top and right spines for the "schematic" look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Labels
ax.set_ylabel(r"Free energy $G$")
ax.set_xlabel(r"Protein conformation $\mathbf{r}$")

# Remove ticks to keep it schematic
ax.set_xticks([])
ax.set_yticks([])

# --- Annotations ---

# 1. Native state (global min)
min_idx = np.argmin(G)
ax.plot(x[min_idx], G[min_idx], "o", color="red", markersize=5, zorder=10)
# Label above
ax.text(
    x[min_idx], G[min_idx] + 3.5, "Native", ha="center", va="bottom", fontweight="bold"
)
# Arrow pointing down to native
ax.annotate(
    "",
    xy=(x[min_idx], G[min_idx] + 0.3),
    xytext=(x[min_idx], G[min_idx] + 3.4),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

# 2. Intermediates (local minima)
# Simple local min finding
local_mins = []
for i in range(1, len(G) - 1):
    if G[i - 1] > G[i] and G[i + 1] > G[i]:
        # Exclude global min area
        if abs(x[i]) > 0.2:
            local_mins.append(i)

# Plot orange dots for intermediates
for idx in local_mins:
    ax.plot(x[idx], G[idx], "o", color="orange", markersize=3.5)

# Label one specific intermediate (e.g., on the left)
left_mins = [i for i in local_mins if x[i] < -0.5]
if left_mins:
    # Pick a prominent one
    idx = min(left_mins, key=lambda i: abs(x[i] - (-1.2)))  # Close to -1.2

    ax.text(x[idx], G[idx] + 2.5, "Intermediates", ha="center", va="bottom")
    ax.annotate(
        "",
        xy=(x[idx], G[idx] + 0.2),
        xytext=(x[idx] - 0.25, G[idx] + 2.4),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

plt.tight_layout()
plt.savefig("figures/folding_funnel_python.pdf")
print("Figure saved to figures/folding_funnel_python.pdf")
