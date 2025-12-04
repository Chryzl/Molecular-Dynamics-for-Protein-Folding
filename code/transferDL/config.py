"""Configuration for Transfer Operator DL experiments."""

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Hyperparameters and settings for the three-phase training protocol."""

    # Architecture
    input_dim: int = 784
    hidden_dim: int = 64
    output_dim: int = 10

    # Phase 1: Minimization (SGD)
    phase1_epochs: int = 20
    phase1_lr_initial: float = 0.01
    phase1_lr_decay: float = 0.95
    phase1_target_accuracy: float = 0.95

    # Phase 2: Equilibration (SGLD)
    phase2_steps: int = 5_000
    sgld_lr: float = 1e-3  # 1e-4 and 5e-4 too low --> not enough exploration
    sgld_noise_scale: float = 1e-2  # σ in method section # 1e-3 too low

    # Phase 3: Production (SGLD)
    phase3_steps: int = 500_000
    save_interval: int = 50  # τ_save

    # Validation
    num_replicas: int = 3
    ck_test_lags: List[int] = None  # Will be set in __post_init__

    # Monitoring
    eval_interval: int = 500  # Check metrics every N steps
    batch_size: int = 128

    # Online Diagnostics (validate SGLD behavior)
    diagnostics_window: int = 1000  # Window for computing statistics
    diagnostics_interval: int = 500  # How often to run diagnostics
    warn_drift_threshold: float = 0.05  # Warn if loss drifts > 5%
    warn_variance_ratio: float = 3.0  # Warn if variance changes > 3x
    min_ess_ratio: float = 0.1  # Minimum effective sample size ratio

    # Paths
    data_dir: str = "./data"
    results_dir: str = "./results"

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.ck_test_lags is None:
            self.ck_test_lags = [1, 2, 5, 10]
