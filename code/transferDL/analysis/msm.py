"""Markov State Model construction and analysis."""

import numpy as np
from typing import Optional


class MarkovStateModel:
    """
    Discretized state space model for kinetics.

    Constructs transition matrix T(τ) from discrete trajectory.
    """

    def __init__(self, n_states: int, lag_time: int):
        """
        Initialize MSM.

        Args:
            n_states: Number of discrete microstates
            lag_time: Lag time τ for transition counting
        """
        self.n_states = n_states
        self.lag = lag_time
        self.transition_matrix = None
        self.stationary_dist = None
        self.cluster_centers = None

    def discretize(self, tica_coords: np.ndarray) -> np.ndarray:
        """
        K-means clustering in TICA space.

        Args:
            tica_coords: Shape (T, n_tica_dims)

        Returns:
            state_sequence: Discrete trajectory S_t, shape (T,)
        """
        pass

    def estimate_transition_matrix(
        self, state_sequence: np.ndarray, reversible: bool = True
    ) -> np.ndarray:
        """
        Estimate transition matrix from discrete trajectory.

        T_ij = N_ij(τ) / Σ_j N_ij(τ)

        Args:
            state_sequence: Discrete trajectory
            reversible: Whether to enforce detailed balance

        Returns:
            transition_matrix: T(τ), shape (n_states, n_states)
        """
        pass

    def compute_stationary_distribution(self) -> np.ndarray:
        """
        Solve π^T T = π^T for largest eigenvalue.

        Returns:
            stationary_dist: π, shape (n_states,)
        """
        pass

    def compute_implied_timescales(self) -> np.ndarray:
        """
        Compute implied timescales from eigenvalues of T.

        t_i = -τ / ln(λ_i)

        Returns:
            timescales: Implied timescales, shape (n_states-1,)
        """
        pass
