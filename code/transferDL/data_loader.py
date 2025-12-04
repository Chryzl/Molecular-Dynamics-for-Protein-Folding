"""Data loading utilities for MNIST with balanced sampling and caching."""

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path


def get_balanced_indices(dataset, samples_per_class=None):
    """
    Get balanced subset indices from dataset.

    Args:
        dataset: PyTorch dataset with .targets attribute
        samples_per_class: Number of samples per class (None = use minimum)

    Returns:
        indices: List of indices for balanced subset
    """
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    # Determine samples per class
    if samples_per_class is None:
        samples_per_class = min([np.sum(targets == c) for c in classes])

    indices = []
    for c in classes:
        class_indices = np.where(targets == c)[0]
        selected = np.random.choice(
            class_indices, size=samples_per_class, replace=False
        )
        indices.extend(selected)

    return indices


def _load_and_cache_mnist(data_dir: str = "./data", split: str = "train"):
    """
    Load MNIST and cache as TensorDataset for faster loading.

    Args:
        data_dir: Directory to store data
        split: 'train' or 'test'

    Returns:
        TensorDataset with all images and labels
    """
    cache_path = Path(data_dir) / f"mnist_{split}_cache.pt"

    # Return from cache if exists
    if cache_path.exists():
        print(f"Loading cached MNIST ({split}) from {cache_path}")
        return torch.load(cache_path)

    print(f"Loading MNIST ({split}) from torchvision (this will be cached)")

    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Standard MNIST preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean, std
        ]
    )

    # Load dataset
    is_train = split == "train"
    dataset = datasets.MNIST(
        root=data_dir, train=is_train, download=True, transform=transform
    )

    # Convert to tensors for faster access
    print(f"Converting to tensor format...")
    images = []
    labels = []
    for img, label in dataset:
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Create TensorDataset
    tensor_dataset = TensorDataset(images, labels)

    # Cache for future runs
    print(f"Caching to {cache_path}")
    torch.save(tensor_dataset, cache_path)

    return tensor_dataset


def get_mnist_dataloaders(batch_size=128, data_dir="./data", balanced=True):
    """
    Create MNIST train and validation dataloaders with balanced classes.
    Uses caching for fast repeated access.

    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load MNIST data
        balanced: Whether to balance classes (recommended for accuracy metric)

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Load cached datasets (or download and cache)
    train_dataset = _load_and_cache_mnist(data_dir, split="train")
    val_dataset = _load_and_cache_mnist(data_dir, split="test")

    # Extract labels for balancing
    train_labels = train_dataset.tensors[1]  # TensorDataset stores (images, labels)
    val_labels = val_dataset.tensors[1]

    # Balance classes if requested
    if balanced:
        # Get balanced indices for training
        train_indices = _get_balanced_indices_from_tensor(train_labels)
        train_dataset = Subset(train_dataset, train_indices)

        # Get balanced indices for validation
        val_indices = _get_balanced_indices_from_tensor(val_labels)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders with pinned memory for fast transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, val_loader


def _get_balanced_indices_from_tensor(labels: torch.Tensor, samples_per_class=None):
    """
    Get balanced subset indices from tensor of labels.

    Args:
        labels: 1D tensor of class labels
        samples_per_class: Number of samples per class (None = use minimum)

    Returns:
        indices: Balanced indices
    """
    labels_np = labels.cpu().numpy()
    classes = np.unique(labels_np)

    # Determine samples per class
    if samples_per_class is None:
        samples_per_class = min([np.sum(labels_np == c) for c in classes])

    indices = []
    for c in classes:
        class_indices = np.where(labels_np == c)[0]
        selected = np.random.choice(
            class_indices, size=samples_per_class, replace=False
        )
        indices.extend(selected)

    return indices


def get_single_batch_loader(dataset, batch_size=128):
    """
    Create a dataloader that returns the same batch repeatedly.
    Useful for SGLD sampling with fixed mini-batch.

    Args:
        dataset: PyTorch dataset
        batch_size: Size of the fixed batch

    Returns:
        loader: Dataloader that cycles through single batch
    """
    indices = np.random.choice(len(dataset), size=batch_size, replace=False)
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return loader
