import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from deeptime.decomposition import TICA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from transferDL.utils import load_trajectory

# ==========================================
# CONFIGURATION
# ==========================================

# Paths to trajectories
CNN_TRAJ_PATH = "/Users/christianmacbook/Documents/MD/results/trajectories/tiny_1k_2mio_25_trajectory.npz"
FFN_TRAJ_PATH = "/Users/christianmacbook/Documents/MD/results/trajectories/32k_1k.npz"

# Output directory for figures
OUTPUT_DIR = "/Users/christianmacbook/Documents/MD/figures"

# Plot settings
FIG_WIDTH = 3.5  # inches (approx IEEE column width)
FONT_SIZE = 8  # pt
DPI = 300

# Analysis settings
TICA_LAGS = [10, 4000]
TICA_DIM = 2
PCA_DIM = 2

# ==========================================
# SCRIPT
# ==========================================


def setup_plotting():
    """Configure matplotlib settings for publication quality."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "figure.titlesize": FONT_SIZE,
            "font.family": "serif",
            "text.usetex": False,  # Set to True if latex is available
        }
    )


def load_data(path):
    """Load trajectory data."""
    print(f"Loading {path}...")
    traj = load_trajectory(path)
    return traj["theta"]


def compute_tica(data, lag):
    """Compute TICA projection."""
    print(f"Computing TICA (lag={lag})...")
    # Convert to float32 for deeptime
    data_f32 = data.astype(np.float32)
    tica = TICA(lagtime=lag, dim=TICA_DIM)
    tica_model = tica.fit(data_f32).fetch_model()
    return tica_model.transform(data_f32)


def compute_pca(data):
    """Compute PCA projection."""
    print(f"Computing PCA...")
    pca = PCA(n_components=PCA_DIM)
    return pca.fit_transform(data)


def plot_tica_comparison(cnn_data, ffn_data):
    """Generate Figure 1: TICA comparison."""
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.8), constrained_layout=True
    )

    models = [("CNN", cnn_data), ("FFN", ffn_data)]

    for row_idx, (model_name, data) in enumerate(models):
        for col_idx, lag in enumerate(TICA_LAGS):
            ax = axes[row_idx, col_idx]

            # Compute TICA
            proj = compute_tica(data, lag)

            # Plot
            ax.plot(proj[:, 0], proj[:, 1], lw=0.5, alpha=0.8)

            # Styling
            if row_idx == 0:
                ax.set_title(f"Lag $\\tau={lag}$")
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nIC 2")
            else:
                ax.set_yticklabels([])

            if row_idx == 1:
                ax.set_xlabel("IC 1")
            else:
                ax.set_xticklabels([])

            ax.grid(True, alpha=0.3)

            # Remove ticks for cleaner look if desired, or keep them
            ax.tick_params(axis="both", which="major", labelsize=6)

    output_path = Path(OUTPUT_DIR) / "tica_comparison.pdf"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()


def plot_pca_comparison(cnn_data, ffn_data):
    """Generate Figure 2: PCA comparison."""
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.5), constrained_layout=True
    )

    models = [("CNN", cnn_data), ("FFN", ffn_data)]

    for idx, (model_name, data) in enumerate(models):
        ax = axes[idx]

        # Compute PCA
        proj = compute_pca(data)

        # Plot
        ax.plot(proj[:, 0], proj[:, 1], lw=0.5, alpha=0.8)

        # Styling
        ax.set_title(f"{model_name}")
        ax.set_xlabel("PC 1")
        if idx == 0:
            ax.set_ylabel("PC 2")
        else:
            ax.set_yticklabels([])

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=6)

    output_path = Path(OUTPUT_DIR) / "pca_comparison.pdf"
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()


def main():
    setup_plotting()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        cnn_theta = load_data(CNN_TRAJ_PATH)
        ffn_theta = load_data(FFN_TRAJ_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Generate plots
    plot_tica_comparison(
        cnn_theta, ffn_theta[:, :5000]
    )  # only plot part of FFN parameters since too large
    plot_pca_comparison(cnn_theta, ffn_theta)


if __name__ == "__main__":
    main()
