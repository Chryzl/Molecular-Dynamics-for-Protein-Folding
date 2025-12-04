"""SGLD optimizer for Bayesian sampling in parameter space."""

import torch
from torch.optim import Optimizer
import math


class SGLD(Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (overdamped).

    Update rule:
        θ_{t+1} = θ_t - ε ∇L(θ_t) + sqrt(2ε) * σ * N(0, I)

    No momentum to ensure reversibility (detailed balance).

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (step size ε)
        noise_scale: Noise magnitude σ
    """

    def __init__(self, params, lr=1e-4, noise_scale=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale: {noise_scale}")

        defaults = dict(lr=lr, noise_scale=noise_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            loss: Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            noise_scale = group["noise_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Gradient descent step
                d_p = p.grad
                p.add_(
                    d_p, alpha=-lr
                )  # TODO, fully random to see if implementation is correct --> We get path in PCA despite fully random???

                # Langevin noise injection: sqrt(2ε) * σ * N(0, I)
                noise_std = math.sqrt(2 * lr) * noise_scale
                noise = torch.randn_like(p) * noise_std
                p.add_(noise)

        return loss
