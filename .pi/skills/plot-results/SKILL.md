---
name: plot-results
description: |
  Generate spectral evolution plots and benchmark correlation figures from
  jlens results.  Covers eigenvalue distribution plots, evolution heatmaps,
  benchmark cross-joins, and the companion analysis scripts.  Use when asked
  to plot, visualize, or analyze collected jlens results.
---

# Plot Results — jlens

Generate figures from collected Jacobian-lens spectral results.

## Available plots

| Script | What it produces |
|---|---|
| `scripts/plot_evolution.py` | Multi-panel spectral evolution dashboard: coherence, α, q_eff, effective rank, spikes, a_coeff, r_norm vs (layer, step) |
| `scripts/plot_eigendist.py` | Eigenvalue distribution + MP fit + power-law tail for a single checkpoint |
| `scripts/plot_benchmark_corr.py` | Benchmark correlation matrix: J-lens metrics × downstream eval scores |
| `scripts/prepare_eval_data.py` | Cross-join eval scores with J-lens metrics → `eval_data/cross_join.npz` |

## Prerequisites

All plots read from `results/step*/stats.h5` and `results/step*/spectra.h5`.
The pipeline must have completed for the checkpoints you want to plot.

## Evolution plots

```bash
uv run python scripts/plot_evolution.py results
```

Output: `results/summary_evolution.png` — multi-panel figure with heatmaps
for each spectral metric across (layer, checkpoint step).

To plot specific layers only:
```bash
uv run python scripts/plot_evolution.py results --layers 0,5,10
```

## Eigenvalue distribution (single checkpoint)

```bash
uv run python scripts/plot_eigendist.py results 143000
```

Output: `results/eigendist_step143000.png` — per-layer panels showing:
- Eigenvalue histogram + MP density overlay
- Power-law tail fit on log-log axes
- Spike threshold line

## Benchmark correlations

### Step 1: Download eval data + cross-join

```bash
uv run python scripts/prepare_eval_data.py
```

Downloads Pythia evaluation JSONs from EleutherAI/pythia, extracts scores
for 8 key benchmarks, cross-joins with J-lens metrics, saves to
`eval_data/cross_join.npz`.

### Step 2: Plot

```bash
uv run python scripts/plot_benchmark_corr.py
```

Output: `eval_data/benchmark_corr.png` — Spearman ρ correlation matrix
(J-lens metrics × benchmarks).

## HDF5 layout

```
results/stepXXXX/
├── metadata.json        # step, n_layers, d_model, n_prompts, config snapshot
├── stats.h5             # per-layer groups: layer_0/layer_1/... with attrs + jbar dataset
│   layer_N/
│     attrs: coherence, a_coeff, r_norm, tr_jtj_mean
│     datasets: jbar (d×d float32)
└── spectra.h5           # per-layer groups with spectral quantities
    layer_N/
      attrs: coherence, a_coeff, r_norm, q_eff, power_law_alpha,
             n_spikes, effective_rank, mp_sigma2, tr_jtj_mean
      datasets: eigenvalues (d,), jbar_gram (d×d)
```

Loading in Python:
```python
from jlens.storage import load_checkpoint_result
result = load_checkpoint_result("results", 143000)
eigvals = result["spectral"]["eigenvalues"]   # list of np.arrays
coherence = result["spectral"]["coherence"]    # list of floats
```
