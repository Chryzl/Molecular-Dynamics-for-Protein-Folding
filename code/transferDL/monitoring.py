"""Training monitoring and metrics tracking."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.decomposition import PCA
import io
from PIL import Image


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
            "phase1": {
                "epoch": [],
                "loss": [],
                "accuracy": [],
                "lr": [],
                "grad_norm": [],
            },
            "phase2": {"step": [], "loss": [], "accuracy": [], "grad_norm": []},
            "phase3": {"step": [], "loss": [], "accuracy": [], "grad_norm": []},
        }

        # Diagnostic storage for SGLD validation
        self.diagnostics = {
            "phase2": {"warnings": []},
            "phase3": {"warnings": []},
        }

        # Trajectory storage for PCA visualization
        self.trajectory_params = []  # Store theta vectors for Phase 3

    def log_phase1(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        lr: float,
        grad_norm: float = None,
    ):
        """Record training metrics during minimization phase."""
        self.metrics["phase1"]["epoch"].append(epoch)
        self.metrics["phase1"]["loss"].append(loss)
        self.metrics["phase1"]["accuracy"].append(accuracy)
        self.metrics["phase1"]["lr"].append(lr)
        if grad_norm is not None:
            self.metrics["phase1"]["grad_norm"].append(grad_norm)

        # TensorBoard logging
        self.writer.add_scalar("Phase1/Loss", loss, epoch)
        self.writer.add_scalar("Phase1/Accuracy", accuracy, epoch)
        self.writer.add_scalar("Phase1/LearningRate", lr, epoch)
        if grad_norm is not None:
            self.writer.add_scalar("Phase1/GradientNorm", grad_norm, epoch)

    def check_phase1_termination(self, accuracy: float) -> bool:
        """
        Check if Phase 1 should terminate.

        Args:
            accuracy: Current validation accuracy

        Returns:
            should_terminate: True if target accuracy reached
        """
        return accuracy >= self.config.phase1_target_accuracy

    def log_phase2(
        self, step: int, loss: float, accuracy: float, grad_norm: float = None
    ):
        """Monitor equilibration (expect stabilization)."""
        self.metrics["phase2"]["step"].append(step)
        self.metrics["phase2"]["loss"].append(loss)
        self.metrics["phase2"]["accuracy"].append(accuracy)
        if grad_norm is not None:
            self.metrics["phase2"]["grad_norm"].append(grad_norm)

        # TensorBoard logging
        self.writer.add_scalar("Phase2/Loss", loss, step)
        self.writer.add_scalar("Phase2/Accuracy", accuracy, step)
        if grad_norm is not None:
            self.writer.add_scalar("Phase2/GradientNorm", grad_norm, step)

    def log_phase3(
        self,
        step: int,
        loss: float,
        accuracy: float,
        theta: np.ndarray = None,
        grad_norm: float = None,
    ):
        """Record production run data and visualize trajectory."""
        self.metrics["phase3"]["step"].append(step)
        self.metrics["phase3"]["loss"].append(loss)
        self.metrics["phase3"]["accuracy"].append(accuracy)
        if grad_norm is not None:
            self.metrics["phase3"]["grad_norm"].append(grad_norm)

        # TensorBoard logging
        self.writer.add_scalar("Phase3/Loss", loss, step)
        self.writer.add_scalar("Phase3/Accuracy", accuracy, step)
        if grad_norm is not None:
            self.writer.add_scalar("Phase3/GradientNorm", grad_norm, step)

        # Store trajectory parameters for PCA visualization
        if theta is not None and False:  # disable for memory and performance
            self.trajectory_params.append(theta)

            # Plot PCA projection periodically
            if (
                len(self.trajectory_params) > 10
                and step % self.config.pca_plot_interval == 0
            ):
                self._plot_pca_projection(step)

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
            if self.metrics["phase1"]["grad_norm"]:
                ax.plot(
                    self.metrics["phase1"]["epoch"],
                    self.metrics["phase1"]["grad_norm"],
                    "m-",
                    linewidth=2,
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Gradient Norm")
                ax.set_title("Phase 1: Minimization - Gradient Norm")
            else:
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
            if self.metrics["phase2"]["grad_norm"]:
                ax.plot(
                    self.metrics["phase2"]["step"],
                    self.metrics["phase2"]["grad_norm"],
                    "m-",
                    linewidth=2,
                )
                ax.set_xlabel("Step")
                ax.set_ylabel("Gradient Norm")
                ax.set_title("Phase 2: Equilibration - Gradient Norm")
            else:
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
            if self.metrics["phase3"]["grad_norm"]:
                steps = np.array(self.metrics["phase3"]["step"])
                grads = np.array(self.metrics["phase3"]["grad_norm"])
                if len(steps) > 1000:
                    indices = np.linspace(0, len(steps) - 1, 1000, dtype=int)
                    steps = steps[indices]
                    grads = grads[indices]
                ax.plot(steps, grads, "m-", alpha=0.6, linewidth=1)
                ax.set_xlabel("Step")
                ax.set_ylabel("Gradient Norm")
                ax.set_title("Phase 3: Production - Gradient Norm")
            else:
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

    def _plot_pca_projection(self, current_step: int):
        """
        Plot PCA projection of trajectory in TensorBoard.

        Args:
            current_step: Current training step for logging
        """
        try:
            # Stack all trajectory parameters
            theta_array = np.array(self.trajectory_params)  # (n_frames, n_params)

            if len(theta_array) < self.config.pca_n_components:
                return  # Not enough samples yet

            # Fit PCA
            pca = PCA(
                n_components=min(self.config.pca_n_components, theta_array.shape[1])
            )
            theta_pca = pca.fit_transform(theta_array)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Scatter plot colored by time (frame index)
            scatter = ax.scatter(
                theta_pca[:, 0],
                theta_pca[:, 1],
                c=np.arange(len(theta_pca)),
                cmap="viridis",
                s=5,
                alpha=0.6,
            )

            ax.set_xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11
            )
            ax.set_ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11
            )
            ax.set_title(
                f"Parameter Space Trajectory (n={len(theta_pca)} frames)", fontsize=12
            )
            ax.grid(alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Time (frame)", fontsize=10)

            plt.tight_layout()

            # Convert to image and log to TensorBoard
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            # TensorBoard expects (C, H, W) format
            if image_array.ndim == 3:
                image_array = np.transpose(image_array, (2, 0, 1))

            self.writer.add_image(
                "Phase3/PCA_Trajectory", image_array, current_step, dataformats="CHW"
            )

            # Log variance explained
            self.writer.add_scalar(
                "Phase3/PCA_PC1_variance",
                pca.explained_variance_ratio_[0],
                current_step,
            )
            self.writer.add_scalar(
                "Phase3/PCA_PC2_variance",
                pca.explained_variance_ratio_[1],
                current_step,
            )

            # Log trajectory spread
            self.writer.add_scalar(
                "Phase3/PCA_PC1_std", np.std(theta_pca[:, 0]), current_step
            )
            self.writer.add_scalar(
                "Phase3/PCA_PC2_std", np.std(theta_pca[:, 1]), current_step
            )

            plt.close(fig)
            buf.close()

        except Exception as e:
            print(f"Warning: PCA plot failed at step {current_step}: {e}")

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
        if len(self.trajectory_params) > 0:
            print(f"\nPhase 3 Statistics:")
            print(f"  Trajectory frames collected: {len(self.trajectory_params)}")
            print(
                f"  PCA plots generated: {len(self.trajectory_params) // self.config.pca_plot_interval}"
            )

        print("=" * 60)

    def close(self):
        """Close TensorBoard writer."""
        self.print_diagnostics_summary()
        self.writer.close()
        print("TensorBoard writer closed")
