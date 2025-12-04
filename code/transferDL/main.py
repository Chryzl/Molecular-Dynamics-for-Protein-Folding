"""Main entry point for Transfer Operator DL experiments."""

import torch
import numpy as np
from pathlib import Path

from config import Config
from data_loader import get_mnist_dataloaders
from training import ThreePhaseTrainer
from utils import load_trajectory

# Skeleton imports for future analysis
# from analysis.tica import TICAAnalyzer
# from analysis.msm import MarkovStateModel
# from analysis.validation import DynamicsValidator


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

    # Analysis phase (skeleton - to be implemented)
    print("\n" + "=" * 60)
    print("ANALYSIS PHASE (TO BE IMPLEMENTED)")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Implement TICA analysis (analysis/tica.py)")
    print("  2. Implement MSM construction (analysis/msm.py)")
    print("  3. Implement validation tests (analysis/validation.py)")
    print("  4. Uncomment analysis code below")

    # Example analysis workflow (currently skeleton)
    """
    print("\nLoading trajectories...")
    trajectories = []
    tica_coords_list = []
    
    for replica_id in range(config.num_replicas):
        traj_path = Path(config.results_dir) / "trajectories" / f"trajectory_replica_{replica_id}.npz"
        traj = load_trajectory(str(traj_path))
        trajectories.append(traj)
        
        # TICA analysis
        print(f"\nReplica {replica_id}: Running TICA...")
        tica = TICAAnalyzer(lag_time=10)
        eigenvalues, eigenvectors = tica.fit(traj['theta'])
        tica_coords = tica.transform(traj['theta'], n_components=2)
        tica_coords_list.append(tica_coords)
        
        # Compute implied timescales
        timescales = tica.compute_implied_timescales(eigenvalues, lag=10)
        print(f"  Top 5 implied timescales: {timescales[:5]}")
        
        # MSM construction
        print(f"\nReplica {replica_id}: Building MSM...")
        msm = MarkovStateModel(n_states=50, lag_time=10)
        state_sequence = msm.discretize(tica_coords)
        msm.estimate_transition_matrix(state_sequence, reversible=True)
        
        stationary = msm.compute_stationary_distribution()
        print(f"  Stationary distribution computed (entropy: {-np.sum(stationary * np.log(stationary + 1e-10)):.4f})")
    
    # Validation
    print("\n" + "="*60)
    print("VALIDATION TESTS")
    print("="*60)
    
    validator = DynamicsValidator()
    
    # Ergodicity test
    print("\n1. Testing ergodicity (replica overlap)...")
    ergodicity_score = validator.test_ergodicity(
        [t['theta'] for t in trajectories],
        tica_coords_list
    )
    print(f"   Ergodicity score: {ergodicity_score:.4f}")
    
    # Detailed balance test
    print("\n2. Testing detailed balance (reversibility)...")
    is_reversible = validator.test_detailed_balance(eigenvalues)
    print(f"   Detailed balance: {'✓ PASS' if is_reversible else '✗ FAIL'}")
    
    # Chapman-Kolmogorov test
    print("\n3. Chapman-Kolmogorov test...")
    ck_results = validator.chapman_kolmogorov_test(
        msm, state_sequence, k_values=config.ck_test_lags
    )
    for k, error in ck_results.items():
        print(f"   k={k}: error = {error:.6f}")
    """

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
