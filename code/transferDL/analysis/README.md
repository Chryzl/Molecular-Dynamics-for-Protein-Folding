# Analysis Pipeline: Transfer Operator Analysis for SGLD Trajectories

## Overview

This analysis pipeline applies molecular dynamics transfer operator methods (TICA, MSMs) to validate SGLD training trajectories and visualize the loss landscape as a free energy surface.

## Structure

```
analysis/
├── analyze_trajectories.py  # Main orchestrator (run this)
├── tica.py                   # TICA analysis with internal tests
├── msm.py                    # MSM construction with CK test
├── validation.py             # Cross-replica ergodicity tests
└── plot_fes.py              # Free energy surface plotting
```

## Usage

After training with `python train.py`, analyze trajectories:

```bash
python analysis/analyze_trajectories.py \
    --results-dir ./results \
    --num-replicas 3 \
    --lag-time 10 \
    --n-tica-components 5 \
    --n-clusters 100 \
    --msm-lag 10
```

## Parameters

- `--results-dir`: Directory containing `trajectories/trajectory_replica_*.npz` files
- `--num-replicas`: Number of training replicas (default: 3)
- `--lag-time`: TICA lag time in trajectory steps (default: 10)
- `--n-tica-components`: Number of TICA components (default: 5)
- `--n-clusters`: Number of MSM microstates (default: 100)
- `--msm-lag`: MSM lag time in saved frames (default: 10)

## Pipeline Steps

### 1. TICA Analysis (`tica.py`)

- **Purpose**: Dimensionality reduction finding slowest collective variables
- **Method**: `deeptime.decomposition.TICA`
- **Tests**:
  - ✓ Eigenvalues positive (detailed balance)
  - ✓ Timescales finite
- **Outputs**:
  - `tica_timescales.png`: Implied timescales by component
  - `tica_projections.png`: Scatter plots of first 2 TICA components per replica

### 2. MSM Construction (`msm.py`)

- **Purpose**: Discretize TICA space and build transition matrix
- **Method**: K-means clustering + `deeptime.markov.msm.MaximumLikelihoodMSM`
- **Tests**:
  - ✓ Chapman-Kolmogorov test (Markovianity)
  - ✓ Stationary distribution normalized
- **Outputs**:
  - `ck_test.png`: CK test validation plot
  - `stationary_distribution.png`: Equilibrium state probabilities

### 3. Validation (`validation.py`)

- **Purpose**: Test ergodicity across replicas
- **Method**: Wasserstein distance between replica distributions in TICA space
- **Tests**:
  - ✓ Replica overlap (ergodicity)
- **Outputs**:
  - `ergodicity_check.png`: TICA space overlap + loss distributions

### 4. Free Energy Surface (`plot_fes.py`)

- **Purpose**: Visualize loss landscape as FES
- **Method**: Map stationary distribution π to TICA grid, compute F = -k_B T ln(π)
- **Outputs**:
  - `free_energy_surface.png`: 2D contour plot of FES

## Output Summary

All results saved to `results/analysis/`:

- `analysis_summary.json`: JSON with all metrics
- 6 PNG figures (see above)

Key metrics reported:

- TICA slowest timescale
- MSM slowest timescale
- Chapman-Kolmogorov test (pass/fail)
- Ergodicity (good/poor overlap)
- Number of free energy minima

## Interpretation

- **TICA timescales**: Larger values → slower dynamics along that direction
- **MSM timescales**: Should be consistent with TICA timescales
- **CK test**: Validates Markovian assumption (estimates match predictions)
- **Ergodicity**: Replicas should overlap (converged sampling)
- **FES**: Low free energy = high probability regions (loss minima)

## Requirements

See `environment.yml`:

- PyTorch 2.0+
- deeptime >=0.4.4
- numpy <2 (compatibility)
- scipy, scikit-learn, matplotlib

## Notes

- Trajectory files must be in `.npz` format with keys: `{steps, theta, loss, accuracy}`
- TICA lag time should be chosen based on autocorrelation decay (see training diagnostics)
- Number of clusters affects MSM resolution vs. statistics tradeoff
- Free energy is computed in dimensionless units (k_B T = 1)
