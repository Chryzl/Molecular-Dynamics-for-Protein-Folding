"""Three-phase training orchestrator for SGLD dynamics."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model import MLP
from optimizers import SGLD
from monitoring import TrainingMonitor
from utils import (
    save_checkpoint,
    load_checkpoint,
    save_trajectory,
    save_initial_weights,
)
from config import Config


class ThreePhaseTrainer:
    """
    Manages the full simulation protocol:
    Phase 1: Minimization (SGD + decay)
    Phase 2: Equilibration (SGLD constant ε)
    Phase 3: Production (SGLD + trajectory saving)
    """

    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        ).to(self.device)

        print(f"Model initialized with {self.model.count_parameters()} parameters")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Monitor (will be re-initialized per replica with correct ID)
        self.monitor = None

        # Trajectory storage for Phase 3
        self.trajectory = []

        # Results directories
        self.checkpoint_dir = Path(config.results_dir) / "checkpoints"
        self.trajectory_dir = Path(config.results_dir) / "trajectories"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, data_loader: DataLoader) -> tuple:
        """
        Compute validation loss and accuracy.

        Args:
            data_loader: Validation dataloader

        Returns:
            (val_loss, val_accuracy): Metrics on validation set
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(data_loader)
        val_accuracy = correct / total if total > 0 else 0.0

        return val_loss, val_accuracy

    def compute_gradient_norm(self) -> float:
        """
        Compute the L2 norm of all gradients in the model.

        Returns:
            grad_norm: L2 norm of all gradients
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return (total_norm) ** 0.5

    def phase1_minimization(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> np.ndarray:
        """
        Phase 1: Standard SGD with exponential LR decay.
        Terminates when accuracy ≥ 95% OR max epochs reached.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader

        Returns:
            theta_min: Minimized parameters as numpy array
        """
        print("\n" + "=" * 60)
        print("PHASE 1: MINIMIZATION (SGD)")
        print("=" * 60)

        optimizer = optim.SGD(self.model.parameters(), lr=self.config.phase1_lr_initial)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config.phase1_lr_decay
        )

        for epoch in range(self.config.phase1_epochs):
            self.model.train()
            train_loss = 0.0
            grad_norm = 0.0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                grad_norm = self.compute_gradient_norm()  # Compute on last batch
                optimizer.step()

                train_loss += loss.item()

            # Evaluate
            val_loss, val_accuracy = self.evaluate(val_loader)
            current_lr = scheduler.get_last_lr()[0]

            # Log metrics
            self.monitor.log_phase1(
                epoch, val_loss, val_accuracy, current_lr, grad_norm
            )

            print(
                f"Epoch {epoch+1}/{self.config.phase1_epochs} - "
                f"Loss: {val_loss:.4f} - Acc: {val_accuracy:.4f} - LR: {current_lr:.6f}"
            )

            # Check termination
            if self.monitor.check_phase1_termination(val_accuracy):
                print(
                    f"\n✓ Target accuracy {self.config.phase1_target_accuracy:.2f} reached!"
                )
                break

            scheduler.step()

        # Save minimized state
        theta_min = self.model.get_flat_params()
        save_checkpoint(
            self.model,
            optimizer,
            self.checkpoint_dir / "phase1_minimized.pt",
            metadata={"epoch": epoch + 1, "accuracy": val_accuracy},
        )

        print(f"Phase 1 complete: Accuracy = {val_accuracy:.4f}")
        return theta_min

    def phase2_equilibration(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Phase 2: SGLD with constant ε for burn-in.
        Allows system to "forget" minimization history.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        print("\n" + "=" * 60)
        print("PHASE 2: EQUILIBRATION (SGLD BURN-IN)")
        print("=" * 60)

        optimizer = SGLD(
            self.model.parameters(),
            lr=self.config.sgld_lr,
            noise_scale=self.config.sgld_noise_scale,
        )

        self.model.train()
        step = 0
        data_iter = iter(train_loader)

        with tqdm(total=self.config.phase2_steps, desc="Equilibration") as pbar:
            while step < self.config.phase2_steps:
                # Get batch (cycle through dataset)
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    inputs, targets = next(data_iter)

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)

                # Periodic evaluation
                if step % self.config.eval_interval == 0:
                    val_loss, val_accuracy = self.evaluate(val_loader)
                    grad_norm = self.compute_gradient_norm()
                    self.monitor.log_phase2(step, val_loss, val_accuracy, grad_norm)
                    self.model.train()

                    pbar.set_postfix(
                        {"loss": f"{val_loss:.4f}", "acc": f"{val_accuracy:.4f}"}
                    )

                # Run equilibration diagnostics
                if (
                    step % self.config.diagnostics_interval == 0
                    and step >= self.config.diagnostics_window
                ):
                    diagnostics = self.monitor.check_phase2_equilibration()
                    if not diagnostics["drift_ok"] or not diagnostics["variance_ok"]:
                        pbar.write(
                            f"Step {step}: Equilibration issues detected (see above)"
                        )

        # Save equilibrated state
        save_checkpoint(
            self.model,
            optimizer,
            self.checkpoint_dir / "phase2_equilibrated.pt",
            metadata={"steps": step},
        )

        print("Phase 2 complete: System equilibrated")

    def phase3_production(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Phase 3: SGLD production run with trajectory saving.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader

        Returns:
            trajectory: List of (step, theta, loss, accuracy) dictionaries
        """
        print("\n" + "=" * 60)
        print("PHASE 3: PRODUCTION (TRAJECTORY SAMPLING)")
        print("=" * 60)

        optimizer = SGLD(
            self.model.parameters(),
            lr=self.config.sgld_lr,
            noise_scale=self.config.sgld_noise_scale,
        )

        self.model.train()
        self.trajectory = []
        step = 0
        data_iter = iter(train_loader)

        with tqdm(total=self.config.phase3_steps, desc="Production") as pbar:
            while step < self.config.phase3_steps:
                # Get batch
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    inputs, targets = next(data_iter)

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)

                # Save trajectory snapshot (use float16 to save memory)
                if step % self.config.save_interval == 0:
                    val_loss, val_accuracy = self.evaluate(val_loader)
                    grad_norm = self.compute_gradient_norm()
                    theta_t = self.model.get_flat_params_fp16()  # float16 for memory

                    self.trajectory.append(
                        {
                            "step": step,
                            "theta": theta_t,
                            "loss": val_loss,
                            "accuracy": val_accuracy,
                        }
                    )

                    self.monitor.log_phase3(
                        step, val_loss, val_accuracy, theta=theta_t, grad_norm=grad_norm
                    )
                    self.model.train()

                    pbar.set_postfix(
                        {
                            "loss": f"{val_loss:.4f}",
                            "acc": f"{val_accuracy:.4f}",
                            "frames": len(self.trajectory),
                        }
                    )

                # Run production diagnostics
                if (
                    step % self.config.diagnostics_interval == 0
                    and step >= self.config.diagnostics_window
                ):
                    diagnostics = self.monitor.check_phase3_sampling()
                    if not diagnostics["sampling_ok"]:
                        pbar.write(f"Step {step}: Sampling issues detected (see above)")

        print(f"Phase 3 complete: {len(self.trajectory)} frames collected")
        return self.trajectory

    def run_replica(
        self, replica_id: int, train_loader: DataLoader, val_loader: DataLoader
    ):
        """
        Execute full 3-phase protocol for one replica.

        Args:
            replica_id: Replica number (for saving)
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        print(f"\n{'#'*60}")
        print(f"REPLICA {replica_id}")
        print(f"{'#'*60}")

        # Phase 1: Minimization
        theta_min = self.phase1_minimization(train_loader, val_loader)
        save_initial_weights(
            self.model, self.checkpoint_dir / f"theta_min_replica_{replica_id}.npy"
        )

        # Phase 2: Equilibration
        self.phase2_equilibration(train_loader, val_loader)

        # Phase 3: Production
        trajectory = self.phase3_production(train_loader, val_loader)

        # Save trajectory
        save_trajectory(
            trajectory, self.trajectory_dir / f"trajectory_replica_{replica_id}.npz"
        )

        # Save metrics and close TensorBoard
        self.monitor.save_metrics(
            Path(self.config.results_dir)
            / "metrics"
            / f"metrics_replica_{replica_id}.json"
        )
        self.monitor.plot_summary(
            Path(self.config.results_dir)
            / "metrics"
            / f"summary_replica_{replica_id}.png"
        )
        self.monitor.close()

        print(f"\n✓ Replica {replica_id} complete!")

    def run_all_replicas(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Execute 3 independent replicas with different random seeds.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        for i in range(self.config.num_replicas):
            # Set different random seed for each replica
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # Reinitialize model for each replica
            self.model = MLP(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
            ).to(self.device)

            # Reinitialize monitor for each replica with correct ID
            self.monitor = TrainingMonitor(self.config, replica_id=i)

            # Run full protocol
            self.run_replica(i, train_loader, val_loader)

        print("\n" + "=" * 60)
        print("ALL REPLICAS COMPLETE")
        print("=" * 60)
