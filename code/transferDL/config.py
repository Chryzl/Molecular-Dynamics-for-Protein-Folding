"""Configuration for Transfer Operator DL experiments."""

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Hyperparameters and settings for the three-phase training protocol."""

    # Architecture
    model_type: str = "FFN"  # "FFN" or "CNN"
    input_dim: int = 784  # For FFN (28*28)
    hidden_dim: int = 32  # For FFN only
    output_dim: int = 10

    # Phase 1: Minimization (SGD)
    phase1_epochs: int = 20
    phase1_lr_initial: float = 0.01
    phase1_lr_decay: float = 0.95
    phase1_target_accuracy: float = 0.95

    # Phase 2: Equilibration (SGLD)
    phase2_steps: int = 10_000
    sgld_lr: float = 5e-4  # 1e-4 and 5e-4 too low --> not enough exploration
    sgld_noise_scale: float = 1e-2  # σ in method section # 1e-3 too low

    # Phase 3: Production (SGLD)
    phase3_steps: int = 20_000_000
    save_interval: int = 1000  # τ_save

    # Validation
    num_replicas: int = 1

    # Monitoring
    eval_interval: int = 500  # Check metrics every N steps
    batch_size: int = 256

    # Online Diagnostics (validate SGLD behavior)
    diagnostics_window: int = 1000  # Window for computing statistics
    diagnostics_interval: int = 500  # How often to run diagnostics
    warn_drift_threshold: float = 0.05  # Warn if loss drifts > 5%
    warn_variance_ratio: float = 3.0  # Warn if variance changes > 3x

    # PCA Trajectory Visualization
    pca_plot_interval: int = 10000  # Plot PCA projection every N steps
    pca_n_components: int = 2  # Number of PCA components to compute

    # Paths
    data_dir: str = "./data"
    results_dir: str = "./results"
