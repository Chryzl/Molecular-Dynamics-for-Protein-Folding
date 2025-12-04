"""Training monitoring and metrics tracking."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    """
    Tracks metrics and handles termination logic for three-phase training.
    Uses both TensorBoard (real-time) and JSON/plots (post-hoc analysis).
    """

    def __init__(
        self, config, save_dir: str = "./results/metrics", replica_id: int = 0
    ):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.replica_id = replica_id

        # Initialize TensorBoard writer
        log_dir = Path(config.results_dir) / "tensorboard" / f"replica_{replica_id}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logging to: {log_dir}")

        # Initialize metric storage (for JSON export and plotting)
        self.metrics = {
            "phase1": {"epoch": [], "loss": [], "accuracy": [], "lr": []},
            "phase2": {"step": [], "loss": [], "accuracy": []},
            "phase3": {"step": [], "loss": [], "accuracy": []},
        }

        # Diagnostic storage for SGLD validation
        self.diagnostics = {
            "phase2": {"warnings": []},
            "phase3": {"warnings": [], "ess": [], "autocorr": []},
        }

    def log_phase1(self, epoch: int, loss: float, accuracy: float, lr: float):
        """Record training metrics during minimization phase."""
        self.metrics["phase1"]["epoch"].append(epoch)
        self.metrics["phase1"]["loss"].append(loss)
        self.metrics["phase1"]["accuracy"].append(accuracy)
        self.metrics["phase1"]["lr"].append(lr)

        # TensorBoard logging
        self.writer.add_scalar("Phase1/Loss", loss, epoch)
        self.writer.add_scalar("Phase1/Accuracy", accuracy, epoch)
        self.writer.add_scalar("Phase1/LearningRate", lr, epoch)

    def check_phase1_termination(self, accuracy: float) -> bool:
        """
        Check if Phase 1 should terminate.

        Args:
            accuracy: Current validation accuracy

        Returns:
            should_terminate: True if target accuracy reached
        """
        return accuracy >= self.config.phase1_target_accuracy

    def log_phase2(self, step: int, loss: float, accuracy: float):
        """Monitor equilibration (expect stabilization)."""
        self.metrics["phase2"]["step"].append(step)
        self.metrics["phase2"]["loss"].append(loss)
        self.metrics["phase2"]["accuracy"].append(accuracy)

        # TensorBoard logging
        self.writer.add_scalar("Phase2/Loss", loss, step)
        self.writer.add_scalar("Phase2/Accuracy", accuracy, step)

    def log_phase3(self, step: int, loss: float, accuracy: float):
        """Record production run data."""
        self.metrics["phase3"]["step"].append(step)
        self.metrics["phase3"]["loss"].append(loss)
        self.metrics["phase3"]["accuracy"].append(accuracy)

        # TensorBoard logging
        self.writer.add_scalar("Phase3/Loss", loss, step)
        self.writer.add_scalar("Phase3/Accuracy", accuracy, step)

    def save_metrics(self, filepath: Optional[str] = None):
        """
        Export all metrics as JSON.

        Args:
            filepath: Path to save metrics (default: save_dir/metrics.json)
        """
        if filepath is None:
            filepath = self.save_dir / "metrics.json"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for phase, phase_data in self.metrics.items():
            serializable_metrics[phase] = {}
            for key, values in phase_data.items():
                serializable_metrics[phase][key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in values
                ]

        with open(filepath, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Metrics saved to {filepath}")

    def plot_summary(self, filepath: Optional[str] = None):
        """
        Generate diagnostic plots for each phase.

        Args:
            filepath: Path to save plot (default: save_dir/training_summary.png)
        """
        if filepath is None:
            filepath = self.save_dir / "training_summary.png"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Phase 1: Minimization
        if self.metrics["phase1"]["epoch"]:
            ax = axes[0, 0]
            ax.plot(
                self.metrics["phase1"]["epoch"],
                self.metrics["phase1"]["loss"],
                "b-",
                linewidth=2,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Phase 1: Minimization - Loss")
            ax.grid(True)

            ax = axes[0, 1]
            ax.plot(
                self.metrics["phase1"]["epoch"],
                self.metrics["phase1"]["accuracy"],
                "g-",
                linewidth=2,
            )
            ax.axhline(
                y=self.config.phase1_target_accuracy,
                color="r",
                linestyle="--",
                label="Target",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Phase 1: Minimization - Accuracy")
            ax.legend()
            ax.grid(True)

        # Phase 2: Equilibration
        if self.metrics["phase2"]["step"]:
            ax = axes[1, 0]
            ax.plot(
                self.metrics["phase2"]["step"],
                self.metrics["phase2"]["loss"],
                "b-",
                linewidth=2,
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Phase 2: Equilibration - Loss")
            ax.grid(True)

            ax = axes[1, 1]
            ax.plot(
                self.metrics["phase2"]["step"],
                self.metrics["phase2"]["accuracy"],
                "g-",
                linewidth=2,
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_title("Phase 2: Equilibration - Accuracy")
            ax.grid(True)

        # Phase 3: Production
        if self.metrics["phase3"]["step"]:
            ax = axes[2, 0]
            # Downsample for plotting if too many points
            steps = np.array(self.metrics["phase3"]["step"])
            losses = np.array(self.metrics["phase3"]["loss"])
            if len(steps) > 1000:
                indices = np.linspace(0, len(steps) - 1, 1000, dtype=int)
                steps = steps[indices]
                losses = losses[indices]
            ax.plot(steps, losses, "b-", alpha=0.6, linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Phase 3: Production - Loss")
            ax.grid(True)

            ax = axes[2, 1]
            steps = np.array(self.metrics["phase3"]["step"])
            accs = np.array(self.metrics["phase3"]["accuracy"])
            if len(steps) > 1000:
                indices = np.linspace(0, len(steps) - 1, 1000, dtype=int)
                steps = steps[indices]
                accs = accs[indices]
            ax.plot(steps, accs, "g-", alpha=0.6, linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_title("Phase 3: Production - Accuracy")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Summary plot saved to {filepath}")

    def check_phase2_equilibration(self, window_size: int = None) -> Dict[str, bool]:
        """
        Check if Phase 2 has equilibrated (no systematic drift).

        Validates:
        1. Loss is not drifting systematically (up or down)
        2. Variance has stabilized

        Args:
            window_size: Number of recent steps to analyze

        Returns:
            diagnostics: Dict with 'drift_ok', 'variance_ok', 'warnings'
        """
        if window_size is None:
            window_size = self.config.diagnostics_window

        losses = np.array(self.metrics["phase2"]["loss"])

        if len(losses) < window_size:
            return {"drift_ok": True, "variance_ok": True, "warnings": []}

        # Split into two halves for comparison
        mid = len(losses) // 2
        first_half = losses[:mid]
        second_half = losses[mid:]

        # Check for systematic drift
        mean_first = np.mean(first_half[-window_size // 2 :])
        mean_second = np.mean(second_half[-window_size // 2 :])
        drift_pct = abs(mean_second - mean_first) / (mean_first + 1e-10)
        drift_ok = drift_pct < self.config.warn_drift_threshold

        # Check variance stabilization
        var_first = np.var(first_half[-window_size // 2 :])
        var_second = np.var(second_half[-window_size // 2 :])
        var_ratio = max(var_first, var_second) / (min(var_first, var_second) + 1e-10)
        variance_ok = var_ratio < self.config.warn_variance_ratio

        warnings = []
        if not drift_ok:
            warning = f"⚠️  Phase 2: Loss drift detected ({drift_pct*100:.2f}% change)"
            warnings.append(warning)
            self.diagnostics["phase2"]["warnings"].append(warning)
            print(f"\n{warning}")

        if not variance_ok:
            warning = f"⚠️  Phase 2: Variance not stabilized (ratio: {var_ratio:.2f}x)"
            warnings.append(warning)
            self.diagnostics["phase2"]["warnings"].append(warning)
            print(f"\n{warning}")

        # Log to TensorBoard
        current_step = self.metrics["phase2"]["step"][-1]
        self.writer.add_scalar("Diagnostics/Phase2_Drift", drift_pct, current_step)
        self.writer.add_scalar(
            "Diagnostics/Phase2_VarianceRatio", var_ratio, current_step
        )

        return {
            "drift_ok": drift_ok,
            "variance_ok": variance_ok,
            "warnings": warnings,
            "drift_pct": drift_pct,
            "var_ratio": var_ratio,
        }

    def check_phase3_sampling(self, window_size: int = None) -> Dict[str, bool]:
        """
        Check if Phase 3 is sampling properly from stationary distribution.

        Validates:
        1. Loss distribution is stable (not drifting)
        2. Effective Sample Size (ESS) is reasonable
        3. Autocorrelation is decaying

        Args:
            window_size: Number of recent steps to analyze

        Returns:
            diagnostics: Dict with validation results and warnings
        """
        if window_size is None:
            window_size = self.config.diagnostics_window

        losses = np.array(self.metrics["phase3"]["loss"])

        if len(losses) < window_size:
            return {"sampling_ok": True, "warnings": []}

        recent_losses = losses[-window_size:]

        # 1. Check for drift (loss should fluctuate around stationary mean)
        first_quarter = recent_losses[: window_size // 4]
        last_quarter = recent_losses[-window_size // 4 :]
        drift_pct = abs(np.mean(last_quarter) - np.mean(first_quarter)) / (
            np.mean(first_quarter) + 1e-10
        )
        drift_ok = drift_pct < self.config.warn_drift_threshold

        # 2. Compute effective sample size (ESS) via autocorrelation
        ess, autocorr_decay = self._compute_ess(recent_losses)
        ess_ratio = ess / len(recent_losses)
        ess_ok = ess_ratio > self.config.min_ess_ratio

        warnings = []
        if not drift_ok:
            warning = f"⚠️  Phase 3: Loss drift detected ({drift_pct*100:.2f}% change) - may not be sampling from stationary distribution"
            warnings.append(warning)
            self.diagnostics["phase3"]["warnings"].append(warning)
            print(f"\n{warning}")

        if not ess_ok:
            warning = f"⚠️  Phase 3: Low ESS ({ess_ratio*100:.1f}%) - high autocorrelation, consider longer lag time"
            warnings.append(warning)
            self.diagnostics["phase3"]["warnings"].append(warning)
            print(f"\n{warning}")

        # Log to TensorBoard
        current_step = self.metrics["phase3"]["step"][-1]
        self.writer.add_scalar("Diagnostics/Phase3_Drift", drift_pct, current_step)
        self.writer.add_scalar("Diagnostics/Phase3_ESS_Ratio", ess_ratio, current_step)
        self.writer.add_scalar(
            "Diagnostics/Phase3_Autocorr_Lag1", autocorr_decay, current_step
        )

        # Store for summary
        self.diagnostics["phase3"]["ess"].append(ess_ratio)
        self.diagnostics["phase3"]["autocorr"].append(autocorr_decay)

        return {
            "drift_ok": drift_ok,
            "ess_ok": ess_ok,
            "sampling_ok": drift_ok and ess_ok,
            "warnings": warnings,
            "drift_pct": drift_pct,
            "ess_ratio": ess_ratio,
            "autocorr_lag1": autocorr_decay,
        }

    def _compute_ess(self, timeseries: np.ndarray, max_lag: int = 1000) -> tuple:
        """
        Compute effective sample size via autocorrelation.

        ESS = N / (1 + 2 * sum(autocorr[lag] for lag > 0))

        Args:
            timeseries: 1D array of values
            max_lag: Maximum lag for autocorrelation

        Returns:
            ess: Effective sample size
            autocorr_lag1: First-order autocorrelation (for decay check)
        """
        n = len(timeseries)

        print(f"Lag time: {min(n, max_lag)} from total samples: {n}")

        timeseries = timeseries - np.mean(timeseries)

        # Compute autocorrelation
        autocorr = (
            np.correlate(timeseries, timeseries, mode="full")[n - 1 :]
            / np.var(timeseries)
            / n
        )
        autocorr = autocorr[: min(max_lag, n)]

        # Sum until autocorrelation becomes negative (standard practice)
        sum_autocorr = 0
        for lag in range(1, len(autocorr)):
            if autocorr[lag] < 0:
                break
            sum_autocorr += autocorr[lag]

        ess = n / (1 + 2 * sum_autocorr)
        autocorr_lag1 = autocorr[1] if len(autocorr) > 1 else 0

        return ess, autocorr_lag1

    def print_diagnostics_summary(self):
        """Print summary of all diagnostics at the end of training."""
        print("\n" + "=" * 60)
        print("SGLD DIAGNOSTICS SUMMARY")
        print("=" * 60)

        # Phase 2 warnings
        phase2_warnings = self.diagnostics["phase2"]["warnings"]
        if phase2_warnings:
            print(f"\nPhase 2 Warnings ({len(phase2_warnings)}):")
            for w in phase2_warnings[-5:]:  # Show last 5
                print(f"  {w}")
        else:
            print("\nPhase 2: ✓ Equilibration successful (no warnings)")

        # Phase 3 warnings
        phase3_warnings = self.diagnostics["phase3"]["warnings"]
        if phase3_warnings:
            print(f"\nPhase 3 Warnings ({len(phase3_warnings)}):")
            for w in phase3_warnings[-5:]:  # Show last 5
                print(f"  {w}")
        else:
            print("\nPhase 3: ✓ Sampling successful (no warnings)")

        # Phase 3 statistics
        if self.diagnostics["phase3"]["ess"]:
            mean_ess = np.mean(self.diagnostics["phase3"]["ess"])
            mean_autocorr = np.mean(self.diagnostics["phase3"]["autocorr"])
            print(f"\nPhase 3 Statistics:")
            print(f"  Mean ESS ratio: {mean_ess*100:.1f}%")
            print(f"  Mean lag-1 autocorr: {mean_autocorr:.3f}")

            if mean_ess < 0.1:
                print(f"  ⚠️  Low ESS suggests high correlation between samples")
                print(f"     Consider: Increase save_interval or run longer")
            else:
                print(f"  ✓ ESS ratio acceptable")

        print("=" * 60)

    def close(self):
        """Close TensorBoard writer."""
        self.print_diagnostics_summary()
        self.writer.close()
        print("TensorBoard writer closed")
