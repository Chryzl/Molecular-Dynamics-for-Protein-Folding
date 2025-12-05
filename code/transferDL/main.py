"""Main entry point for Transfer Operator DL experiments."""

import torch
import numpy as np
from pathlib import Path

from config import Config
from data_loader import get_mnist_dataloaders
from training import ThreePhaseTrainer
from utils import load_trajectory


def main():
    """Execute three-phase SGLD training and analysis."""

    # Configuration
    config = Config()

    print("=" * 60)
    print("TRANSFER OPERATOR ANALYSIS OF NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(
        f"  Architecture: {config.input_dim} → {config.hidden_dim} → {config.output_dim}"
    )
    print(
        f"  Phase 1: {config.phase1_epochs} epochs (target acc: {config.phase1_target_accuracy})"
    )
    print(f"  Phase 2: {config.phase2_steps} steps (equilibration)")
    print(f"  Phase 3: {config.phase3_steps} steps (production)")
    print(f"  SGLD: lr={config.sgld_lr}, noise={config.sgld_noise_scale}")
    print(f"  Replicas: {config.num_replicas}")
    print(f"  Save interval: {config.save_interval} steps")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create data loaders (balanced classes)
    print("\nLoading MNIST dataset (balanced classes)...")
    train_loader, val_loader = get_mnist_dataloaders(
        batch_size=config.batch_size, data_dir=config.data_dir, balanced=True
    )

    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    trainer = ThreePhaseTrainer(config, device=device)

    # Run all replicas
    trainer.run_all_replicas(train_loader, val_loader)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - TRAJECTORIES SAVED")
    print("=" * 60)
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir={config.results_dir}/tensorboard")


if __name__ == "__main__":
    main()
