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
| `scripts/plot_circle_probe.py` | Circle probe (Qwen3.6): loop-vs-open-path score vs layer, per-head heatmaps, best units in PCA. `--scores /data/jlens/qwen3.6-27b/circle_probe/scores.csv --probe .../probe.h5` (run on a compute node) |
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

## Pythia-12B raw-spectrum dashboard

```bash
uv run python scripts/plot_spectra_dashboard.py
```

Writes `figures/pythia-12b/spectra_dashboard.html`. It compares the
wikitext- and random-token-averaged runs at every completed common checkpoint.
Use its checkpoint slider or Play control to animate training, and its layer
menu to focus on a depth band. Both panels show the raw positive eigenvalues
of `Jbar_l^T Jbar_l` against descending rank on log axes; rank samples are
reduced only for browser rendering.


## Pythia-12B spectrum histogram dashboard

```bash
uv run python scripts/plot_spectra_histogram_dashboard.py
```

Writes `figures/pythia-12b/spectra_histogram_dashboard.html`. It has the same
checkpoint animation, corpus comparison, layer colors, and depth-band selector
as the rank-spectrum dashboard, but displays superposed log-binned
probability-density histogram curves for each layer. It uses only the stored
raw eigenvalues; it does not fit or overlay a Marchenko-Pastur curve.


Histogram eigenvalues are raw (unnormalized), binned on a fixed shared log grid (1e-6..1e3). The output uses globally fixed axis limits across all animation
frames, and playback goes through `scripts/dashboard_playback.py` (custom JS, vertical log-space interpolation, no Plotly frames); this
prevents either scale drift or cross-layer trace mixing during playback.


## Separate corpus dashboards

Generate a standalone dashboard for either corpus with
`plot_spectrum_dashboard.py` (rank spectra) or
`plot_spectrum_histogram.py` (empirical densities):

```bash
uv run python scripts/plot_spectrum_dashboard.py \
  --results-dir /data/jlens/pythia-12b-deduped \
  --label Wikitext-averaged \
  --output figures/pythia-12b/spectra_wikitext.html
uv run python scripts/plot_spectrum_histogram.py \
  --results-dir /data/jlens/pythia-12b-deduped-random \
  --label Random-token-averaged \
  --output figures/pythia-12b/spectra_histogram_random.html
```

The generated files are `spectra_wikitext.html`, `spectra_random.html`,
`spectra_histogram_wikitext.html`, and `spectra_histogram_random.html`.


For animated rank spectra, retain non-positive numerical eigenvalues at the
smallest positive spectral value rather than filtering them out: every frame
must preserve the same sampled rank coordinates for vertical interpolation.


## OLMo-3-1125-32B (normalized eigenvalues)

`olmo3_analysis/` holds the OLMo-3 copies of the fit/plot scripts. Unlike the
Pythia scripts in `scripts/`, they work in normalized units λ/d_model: every
eigenvalue read goes through `normalized_eigenvalues()` and the fit window
`BINS` and histogram window are the raw [1e-6, 1e3] divided by `D_MODEL`
(5120, checked against the `d_model` attr in each `spectra.h5`). Never divide
eigenvalues without also moving these windows.

The three-component mixture fit runs in two passes: pass 1 is a multistart
that also varies the MP shape q (a single q start gets trapped at the q bound
on the very narrow top-layer spectra); pass 2 (`--seed-dir`) restarts every
(step, layer) from the pass-1 optima at that step and its neighbouring steps
and keeps the best, so it is never worse than pass 1 and smooths the
composition across steps. Each pass takes ~2-6 min as a 61-task array.

```bash
cd olmo3_analysis
mv results/olmo3_3mix_pass1 results/olmo3_3mix_parts <somewhere>  # fit array skips steps with an existing CSV
P1=$(sbatch --parsable olmo3_fit_array.sh results/olmo3_3mix_pass1)          # pass 1: 16-start multistart, 61 tasks
P2=$(sbatch --parsable --dependency=afterok:$P1 olmo3_fit_array.sh \
       results/olmo3_3mix_parts results/olmo3_3mix_pass1)                     # pass 2: refine from neighbouring steps
sbatch --dependency=afterok:$P2 olmo3_combine_plot.sh     # overlays + heatmap -> figures/
sbatch olmo3_histogram.sh                                 # figures/spectra_histogram_wikitext.html
cp figures/* ../figures/olmo3/
```

Jobs are pinned with `--nodelist` to a single node (shared account); change it
if that node is busy. Tests: `tests/test_olmo3_normalization.py` (run on a
compute node — the fits are very slow on the login node).
