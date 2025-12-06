"""Simple MLP and CNN architectures for MNIST classification."""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


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


class CNN(nn.Module):
    """
    Tiny CNN for MNIST.
    Architecture: Conv → MaxPool → Conv → Flatten → FC
    """

    def __init__(self, output_dim=10):
        super().__init__()
        self.output_dim = output_dim

        # Conv block 1: 1 → 4 channels, 3×3 kernel
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28×28 → 14×14

        # Conv block 2: 4 → 8 channels, 3×3 kernel
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14×14 → 7×7

        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 7×7 → 3×3

        # Flatten: 8 * 3 * 3 = 72
        self.fc1 = nn.Linear(8 * 3 * 3, output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
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


def build_model(model_type: str, config) -> Union[MLP, CNN]:
    """
    Factory function to instantiate the correct model based on config.

    Args:
        model_type: "FFN" or "CNN"
        config: Config object containing hyperparameters

    Returns:
        Initialized model (MLP or CNN)

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "FFN":
        return MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )
    elif model_type == "CNN":
        return CNN(output_dim=config.output_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'FFN' or 'CNN'.")
