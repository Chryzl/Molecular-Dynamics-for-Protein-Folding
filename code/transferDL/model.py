"""Simple MLP architecture for MNIST classification."""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    Simple feedforward network for MNIST.
    Architecture: 784 → 64 → 10 (~50K parameters)
    """

    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        # Flatten image: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_flat_params(self) -> np.ndarray:
        """
        Return θ as 1D numpy array for trajectory storage.

        Returns:
            theta: All parameters concatenated into single vector
        """
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def get_flat_params_fp16(self) -> np.ndarray:
        """
        Return θ as 1D numpy array in float16 for memory-efficient storage.

        Returns:
            theta: All parameters concatenated into single vector (float16)
        """
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().astype(np.float16).flatten())
        return np.concatenate(params)

    def set_flat_params(self, theta: np.ndarray):
        """
        Restore θ from 1D vector.

        Args:
            theta: Flattened parameter vector
        """
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(
                torch.from_numpy(theta[offset : offset + numel]).view_as(param)
            )
            offset += numel

        if offset != len(theta):
            raise ValueError(
                f"Parameter size mismatch: expected {offset}, got {len(theta)}"
            )

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
