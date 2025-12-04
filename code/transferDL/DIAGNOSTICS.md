# Online SGLD Diagnostics Guide

## Overview

The training pipeline now includes **real-time validation** of SGLD dynamics to detect issues early, before downstream analysis breaks.

## What Gets Checked

### Phase 2: Equilibration Diagnostics

**Goal**: Verify the system has "forgotten" the minimization and reached the stationary distribution of the SGLD process.

**Checks**:

1. **Drift Detection** (`check_phase2_equilibration`)

   - Compares mean loss between first and second half of recent window
   - **Warning if**: Drift > 5% (systematic increase/decrease)
   - **Indicates**: Not yet equilibrated, still transitioning

2. **Variance Stabilization**
   - Compares variance between halves
   - **Warning if**: Variance ratio > 3x
   - **Indicates**: System not stabilized, still settling

**Run every**: 500 steps (configurable via `diagnostics_interval`)

### Phase 3: Production Diagnostics

**Goal**: Verify we're properly sampling from the stationary distribution, not trapped or drifting.

**Checks**:

1. **Stationarity** (`check_phase3_sampling`)

   - Compares loss distribution across time
   - **Warning if**: Mean drift > 5% across window
   - **Indicates**: Not sampling from stationary distribution

2. **Effective Sample Size (ESS)**

   - Computes: `ESS = N / (1 + 2 * Σ autocorr[lag])`
   - **Warning if**: ESS ratio < 10%
   - **Indicates**: Samples highly correlated, need longer lag or more steps

3. **Autocorrelation Decay**
   - Tracks lag-1 autocorrelation
   - **Good**: Decays quickly to near-zero
   - **Bad**: Stays high, samples are not independent

## How to Use

### During Training

Warnings appear automatically in console:

```
⚠️  Phase 2: Loss drift detected (6.2% change)
⚠️  Phase 3: Low ESS (8.5%) - high autocorrelation, consider longer lag time
```

### In TensorBoard

View real-time diagnostic plots:

```bash
tensorboard --logdir=./results/tensorboard
```

Navigate to `Diagnostics/` tab to see:

- `Phase2_Drift`: Should trend toward 0
- `Phase2_VarianceRatio`: Should stabilize below 3.0
- `Phase3_Drift`: Should stay near 0
- `Phase3_ESS_Ratio`: Should stay above 0.1
- `Phase3_Autocorr_Lag1`: Should be low (<0.5)

### After Training

Run comprehensive diagnostics:

```bash
python visualize_diagnostics.py --replica 0 --phase phase3
```

Generates:

- Loss trajectory plot (check for stationarity)
- Autocorrelation plot (check for mixing)
- Statistical summary (drift, ESS, etc.)

## Interpreting Results

### ✅ Good Signs

- **Phase 2**:

  - Drift < 5%
  - Variance ratio < 3x
  - Loss stabilizes around constant mean

- **Phase 3**:
  - Drift < 5% throughout
  - ESS ratio > 10%
  - Autocorrelation decays within ~50 lags
  - Loss distributions overlap across time

### ⚠️ Warning Signs

- **High Drift** → Not sampling from stationary distribution

  - **Fix**: Run Phase 2 longer, check SGLD parameters

- **Low ESS** → Samples too correlated

  - **Fix**: Increase `save_interval`, run longer, increase noise

- **High Autocorrelation** → Slow mixing

  - **Fix**: Increase noise scale, use different optimizer

- **Distributions Don't Overlap** → Non-stationary
  - **Fix**: Check for bugs, validate SGLD implementation

## Configuration

Adjust thresholds in `config.py`:

```python
# Online Diagnostics
diagnostics_window: int = 1000        # Window for statistics
diagnostics_interval: int = 500       # How often to check
warn_drift_threshold: float = 0.05    # Max drift (5%)
warn_variance_ratio: float = 3.0      # Max variance change
min_ess_ratio: float = 0.1            # Min ESS (10%)
```

## What This Prevents

Without these checks, you might:

1. Build TICA/MSM on non-equilibrated data → Wrong eigenfunctions
2. Use highly correlated samples → Overestimate convergence
3. Sample from transient distribution → Wrong free energies
4. Violate Markovian assumption → Invalid MSM

With checks, you catch issues **during training** instead of **after hours of analysis**.

## Technical Details

### ESS Computation

```python
# Autocorrelation
autocorr[lag] = <x(t) * x(t+lag)> / var(x)

# Sum until negative
integrated_autocorr = sum(autocorr[lag>0] while autocorr > 0)

# Effective sample size
ESS = N / (1 + 2 * integrated_autocorr)
```

### Drift Detection

```python
# Compare first vs last quarter
first_quarter_mean = mean(losses[:N//4])
last_quarter_mean = mean(losses[-N//4:])

drift_pct = |last - first| / first * 100

if drift_pct > threshold:
    warn("Non-stationary!")
```

## References

- Geyer (1992): "Practical Markov Chain Monte Carlo" - ESS via autocorrelation
- Mandt et al. (2017): "SGD as Approximate Bayesian Inference" - SGLD stationarity
- Betancourt (2018): "A Conceptual Introduction to HMC" - Diagnostics for MCMC
