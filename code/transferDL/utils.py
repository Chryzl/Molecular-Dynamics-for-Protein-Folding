"""Utility functions for checkpointing and trajectory management."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def save_checkpoint(model, optimizer, filepath: str, metadata: Optional[Dict] = None):
    """
    Save model state + optimizer state + metadata.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to save checkpoint
        metadata: Optional dictionary with additional info (epoch, accuracy, etc.)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):
    """
    Restore training state from checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional PyTorch optimizer to load state into

    Returns:
        metadata: Dictionary with checkpoint metadata (if present)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metadata = checkpoint.get("metadata", {})
    print(f"Checkpoint loaded from {filepath}")

    return metadata


def save_trajectory(trajectory_list: List[Dict], filepath: str):
    """
    Save trajectory as compressed NumPy archive.

    Args:
        trajectory_list: List of dictionaries with keys:
            - 'step': int
            - 'theta': np.ndarray (parameter vector)
            - 'loss': float
            - 'accuracy': float
        filepath: Path to save trajectory (.npz file)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not trajectory_list:
        raise ValueError("Cannot save empty trajectory")

    # Convert list of dicts to dict of arrays
    steps = np.array([t["step"] for t in trajectory_list])
    theta = np.array(
        [t["theta"] for t in trajectory_list], dtype=np.float16
    )  # Ensure float16
    losses = np.array([t["loss"] for t in trajectory_list])
    accuracies = np.array([t["accuracy"] for t in trajectory_list])

    # Calculate memory savings
    mem_mb = theta.nbytes / (1024 * 1024)

    np.savez_compressed(
        filepath, steps=steps, theta=theta, loss=losses, accuracy=accuracies
    )

    print(
        f"Trajectory saved to {filepath} ({len(trajectory_list)} frames, {mem_mb:.1f} MB theta)"
    )


def load_trajectory(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load trajectory from disk.

    Args:
        filepath: Path to trajectory file (.npz)

    Returns:
        trajectory: Dictionary with keys 'steps', 'theta', 'loss', 'accuracy'
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory not found: {filepath}")

    data = np.load(filepath)
    trajectory = {
        "steps": data["steps"],
        "theta": data["theta"],
        "loss": data["loss"],
        "accuracy": data["accuracy"],
    }

    print(f"Trajectory loaded from {filepath} ({len(trajectory['steps'])} frames)")
    return trajectory


def save_initial_weights(model, filepath: str):
    """
    Save initial weights from Phase 1 minimization.

    Args:
        model: PyTorch model
        filepath: Path to save weights
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    theta_init = model.get_flat_params()
    np.save(filepath, theta_init)

    print(f"Initial weights saved to {filepath} ({len(theta_init)} parameters)")


def load_initial_weights(filepath: str) -> np.ndarray:
    """
    Load initial weights for replication.

    Args:
        filepath: Path to saved weights

    Returns:
        theta_init: Flattened parameter vector
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Initial weights not found: {filepath}")

    theta_init = np.load(filepath)
    print(f"Initial weights loaded from {filepath} ({len(theta_init)} parameters)")

    return theta_init
