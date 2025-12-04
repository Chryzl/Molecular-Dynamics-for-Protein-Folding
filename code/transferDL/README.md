# Transfer Operator Analysis of Neural Network Training

This project applies molecular dynamics analysis techniques (TICA, MSMs) to study the dynamics of neural network training in parameter space.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate transferDL
```

## Usage

Run the full three-phase training protocol:

```bash
python main.py
```

### Monitor Training with TensorBoard

In a separate terminal, launch TensorBoard:

```bash
tensorboard --logdir=./results/tensorboard
```

Then open your browser to `http://localhost:6006` to view:

- Real-time loss and accuracy curves
- Separate tabs for Phase 1, 2, and 3
- Comparison across replicas

This will:

1. **Phase 1 (Minimization)**: Train with SGD until 95% accuracy
2. **Phase 2 (Equilibration)**: Run SGLD for 5K steps to forget minimization history
3. **Phase 3 (Production)**: Sample 500K steps, saving trajectory every 50 steps
4. Repeat for 3 independent replicas

## Project Structure

```
transferDL/
├── config.py           # Hyperparameters
├── model.py            # MLP architecture
├── optimizers.py       # SGLD implementation
├── data_loader.py      # MNIST loaders (balanced)
├── training.py         # Three-phase orchestrator
├── monitoring.py       # Metrics tracking
├── utils.py            # Checkpointing & I/O
├── main.py             # Entry point
└── analysis/           # Analysis tools (skeleton)
    ├── tica.py         # Time-lagged ICA
    ├── msm.py          # Markov State Models
    └── validation.py   # Ergodicity, CK tests
```

## Results

Results are saved in `./results/`:

- `checkpoints/`: Model states after each phase
- `trajectories/`: Parameter trajectories (`.npz` format)
- `metrics/`: Loss/accuracy time series (JSON) + summary plots (PNG)
- `tensorboard/`: Real-time TensorBoard logs (view with `tensorboard --logdir=./results/tensorboard`)

## Online Diagnostics

The training pipeline includes **real-time validation** of SGLD dynamics to catch issues early:

### Phase 2 (Equilibration) Diagnostics:

- **Drift Detection**: Warns if loss is systematically increasing/decreasing (>5% change)
- **Variance Stabilization**: Checks if loss variance has stabilized (ratio <3x between halves)

### Phase 3 (Production) Diagnostics:

- **Stationarity Check**: Monitors if loss distribution is stable (not drifting)
- **Effective Sample Size (ESS)**: Computes ESS via autocorrelation to detect over-correlated samples
- **Autocorrelation Decay**: Tracks lag-1 autocorrelation to assess mixing

### Warning System:

- ⚠️ Warnings printed to console when thresholds exceeded
- All diagnostics logged to TensorBoard under `Diagnostics/` tab
- Summary report at end of each replica

**Configuration** (in `config.py`):

```python
diagnostics_window: int = 1000        # Window size for statistics
diagnostics_interval: int = 500       # How often to check
warn_drift_threshold: float = 0.05    # Max acceptable drift (5%)
warn_variance_ratio: float = 3.0      # Max variance change
min_ess_ratio: float = 0.1            # Minimum ESS (10%)
```

### Post-Training Diagnostics:

After training, visualize diagnostics for any replica:

```bash
python visualize_diagnostics.py --replica 0 --phase phase3
```

This generates:

- **Loss trajectory** with rolling mean/std and distribution evolution
- **Autocorrelation function** to assess mixing quality
- **Summary statistics** (drift, ESS, etc.)

## Analysis (To Be Implemented)

The analysis module provides skeletons for:

- **TICA**: Identify slow collective variables
- **MSM**: Discretize state space, compute kinetics
- **Validation**: Ergodicity, detailed balance, Chapman-Kolmogorov tests

## Method

This implementation follows the protocol described in the paper's Method section:

1. **Time-homogeneous dynamics**: Constant SGLD step size (no annealing)
2. **Reversibility**: Overdamped Langevin (no momentum)
3. **Ergodicity**: Validated via replica overlap in TICA space
4. **Markovianity**: Validated via Chapman-Kolmogorov test

## References

- Mandt et al. (2017): "Stochastic Gradient Descent as Approximate Bayesian Inference"
- Pérez-Hernández et al. (2013): "Identification of slow molecular order parameters for Markov model construction"
- Prinz et al. (2011): "Markov models of molecular kinetics"
