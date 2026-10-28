# PROGRESS.md — J-Lens: Spectral Evolution of the Jacobian Workspace Across Pythia-160M Training
We record here significant results (positive or negative). The level of details
is adapted to match the importance of the result.

## Overview

This project measures the **Jacobian lens** $\bar{J}_\ell = \mathbb{E}[\partial h_{\text{final}} / \partial h_\ell]$ across 153 Pythia-160M-deduped pretraining checkpoints (step 1—143,000, 1000 wikitext prompts each), using the `anthropics/jacobian-lens` reference implementation. We cross-correlate the resulting spectral metrics with EleutherAI benchmark evaluations at 26 overlapping checkpoints.

**Sweep complete:** 153/153 checkpoints done. Overlap sweep complete (26/27 checkpoints). WMT eval complete (153/153). 71/71 tests passing. Publication-quality plots and metric dictionary exist.

---

## Key Results

### 1. The Identity Coefficient $a_0$ is the best predictor of downstream performance

$$a_\ell = \operatorname{tr}(\bar{J}_\ell) / d$$

- $a_0$ drops monotonically from 1.00 (step 1) to 0.17 (step 143K).
- **$a_0$ × ARC-Easy: Spearman $\rho = -0.90^{**}$** — the strongest single correlation in the entire dataset.
- $a_0$ is a scalar readout of how much "computation" the model has learned: at init, transport is $J \approx I$ (pure skip connection, $a = 1$). As training proceeds, off-diagonal structure accumulates, and the identity component shrinks.
- No other metric (coherence, effective rank, power-law exponent, spike count) matches this predictive strength.

### 2. $\kappa_\ell$ (transport coherence) is NOT a monotonic order parameter

$$\kappa_\ell = \frac{\|\bar{J}_\ell\|_F^2}{\mathbb{E}[\|J_\ell\|_F^2]}$$

**The paradox:** $\kappa_0$ *falls* during training (0.94 → 0.37), the opposite of naive workspace predictions.

**Resolution:** $\kappa_\ell$ conflates two distinct phenomena:
- **Trivial agreement at initialization** ($\kappa \approx 1$): all per-prompt Jacobians are $J \approx I$ because the model hasn't learned anything. Agreement is vacuous.
- **Structured computation at convergence** (lower $\kappa$): per-prompt Jacobians develop genuine prompt-dependent structure in early layers, making the mean less representative.

The real workspace signal is the **depth profile** $\kappa(\ell)$: flat at init (~0.94 everywhere), graded at convergence (0.37 in sensory layers → 0.97 in motor layers). The gradient is the workspace, not any individual $\kappa_\ell$ value.

**Better workspace diagnostics** (future work):
- Fluctuation-to-coherent ratio: $\operatorname{tr}(\Sigma_\ell) / \|\bar{J}_\ell\|_F^2 = (1-\kappa_\ell)/\kappa_\ell$
- Cross-prompt eigenvector overlap: do the top-$k$ eigenvectors of $\bar{J}_\ell$ agree across independent prompt batches?

### 3. The power-law exponent $\alpha \approx 0.44$ is near-universal

For a product of $N$ *independent* random matrices, the small-$\lambda$ CDF follows a Fuss-Catalan distribution with $\alpha_{\text{null}} = 1/(N+1)$. For Pythia-160M ($N = 12$ layers), this predicts $\alpha \approx 0.08$.

**We measure $\alpha \approx 0.44$** at Layer 0 at late training — matching the Qwen3.5-4B measurement despite a 25× difference in model size. This means the effective free-factor count is $\approx 1.2$: the chain of 12 per-block Jacobians has the small-singular-value statistics of roughly **one** matrix. The factors are strongly aligned into an approximately shared singular basis.

$\alpha$ correlates positively with PIQA ($\rho = +0.82$) and LAMBADA ($\rho = +0.77$). More alignment = better performance.

### 4. Effective rank is non-monotonic across training

$$\text{eff\_rank} = \exp(-\sum p_k \ln p_k), \quad p_k = \lambda_k / \sum \lambda_j$$

- Step 1: 242 → Step 128: 163 (identity collapse) → Step 1K: 241 (structuration rebound) → Step 143K: 154 (late plateau)
- The model first collapses the isotropic init spectrum into a few dominant modes, then expands the basis as structured features emerge, then prunes again.
- **Not predicted by any existing theory.** This "drainage" pattern is a novel empirical finding.

### 5. Spike population collapses abruptly at step ~2000

$n_{\text{spikes},0}$ drops from ~143 (step 1) to ~1-4 (post-step 2000). The most threshold-gated behavioral change in any metric. Deep-layer spikes (layer 5) correlate *positively* with performance ($\rho = +0.86$ with LAMBADA), while shallow-layer spikes correlate negatively — suggesting different computational roles.

### 6. Deep layers ($\ell \geq 5$) stay near-isometric throughout training

$a_\ell \approx 1.0$, $\kappa_\ell > 0.96$ for layers 5-10 at all training stages. The "motor band" remains near-identity regardless of what the early layers learn. This is consistent with the workspace picture: late layers form a stable readout channel.

---

## Four Training Phases at Layer 0

| Phase | Steps | $a_0$ | $\kappa_0$ | Eff. rank | $n_{\text{spikes}}$ |
|-------|-------|-------|------------|-----------|---------------------|
| 1. Near-identity | 1–64 | 1.00–0.95 | 0.95–0.93 | 242–240 | 164–143 |
| 2. Identity collapse | 128–512 | 0.84–0.62 | 0.93–0.91 | 112–163 | 111–63 |
| 3. Structuration | 1K–20K | 0.42–0.17 | 0.82–0.72 | 209–182 | 35–7 |
| 4. Late plateau | 30K–143K | 0.18–0.17 | 0.70–0.37 | 173–125 | 7–0 |

**Anomaly:** Step 107,000 shows a sudden $\kappa_0$ drop (0.51 → 0.03), possibly a corrupted checkpoint or convergence instability.

---

## Correlation Summary: J-Lens × Benchmarks

Top Spearman $\rho$ (26 matched checkpoints, all $|\rho| > 0.80$ significant at $p < 0.01$):

| Metric | Benchmark | $\rho$ |
|--------|-----------|--------|
| $a_0$ | ARC-Easy | **−0.901** |
| $n_0$ (spikes L0) | SciQ | −0.894 |
| $n_0$ | ARC-Easy | −0.891 |
| $a_0$ | SciQ | −0.886 |
| $\mathbb{E}[\text{tr} J^\top J]_0$ | SciQ | −0.881 |
| $\mathbb{E}[\text{tr} J^\top J]_0$ | ARC-Easy | −0.880 |
| $\kappa_0$ | ARC-Easy | −0.840 |
| $n_5$ (spikes L5) | LAMBADA | **+0.859** |
| $n_5$ | PIQA | +0.826 |
| $n_5$ | SciQ | +0.823 |
| $\bar{\alpha}$ (mean) | PIQA | +0.820 |

**Key pattern:** Layer 0 metrics anti-correlate with performance (the model gets better as early-layer transport departs from identity). Layer 5 spikes correlate positively (deep-layer dominant directions serve a routing function).

---

## WMT14 fr-en Translation Analysis

We ran inference on all 153 checkpoints on WMT14 French→English (3003 examples). Pythia-160M is monolingual English and never learns to translate: BLEU ≈ 0–0.4 everywhere. But chrF (character n-gram overlap, 0–11.2) and TER (edit distance, 388→159) capture increasing English fluency.

**Key finding:** Translation metrics correlate 1.5× weaker with J-lens metrics than comprehension benchmarks:

| Task type | Best \|ρ\| | Best \|r\| | Metric |
|---|---|---|---|
| Comprehension (ARC-Easy) | 0.898 | 0.894 | Identity a L0 |
| Comprehension (WinoGrande) | 0.899 | 0.896 | Spike count L0 |
| **Translation (BLEU)** | **0.613** | **0.493** | **Power-law α L0** |
| **Translation (TER)** | **0.579** | **0.494** | **Coherence κ L0** |

This supports the hypothesis that the Jacobian workspace is **working memory**, not a computation site — translation requires cross-lingual computation elsewhere.

---

## Cross-Prompt Eigenvector Overlap

Measured on 27 eval-matched checkpoints with 1000+1000 prompts (seed 42 vs seed 123). Uses Grassmann overlap: $(1/k) \operatorname{tr}(U_A^\top U_B U_B^\top U_A)$ with $k=100$.

**Key result: Overlap ≈ 1.0 across ALL training stages.** Minimum 0.9942 (step 123K), maximum 1.0000 (step 32). This was unexpected given $\kappa_0$ falls from 0.94→0.37.

**Implication:** The dominant eigenspace of $\bar{J}_0$ is nearly identical across independent prompt batches, even when coherence κ is low. Low κ ≠ subspace disagreement — it means the *norm* of J fluctuates per prompt, but the directions stay aligned. The workspace is a **stable subspace** from initialization onward. What changes across training is not its geometry but its energy content (FCR rising, identity component falling).

### FCR (Fluctuation-to-Coherent Ratio)

$\operatorname{FCR} = \operatorname{tr}(\Sigma) / \|\bar{J}\|_F^2 = (1-\kappa)/\kappa$

- Step 1: FCR = 0.065 → fluctuation is 6.5% of coherent energy
- Step 143K: FCR = 1.43 → fluctuation exceeds coherent energy

FCR rises monotonically alongside training, driven entirely by $\kappa$ falling, since the subspace itself doesn't change.

---

## Architecture & Implementation Notes

- **Jacobian estimator:** `anthropics/jacobian-lens` (v0.1.3), installed via `uv add` with Git dependency. Bridge module `jlens/_bridge.py` handles namespace collision with local `jlens` package.
- **Pythia layout:** Pythia-160m uses `lm_head` instead of `embed_out`. Explicit `Layout(lm_head='lm_head')` in `_wrap_model()`.
- **Layer constraint:** `source_layers < target_layer = n_layers - 1`. Runner defaults to `list(range(n_layers - 1))` (layers 0–10). Layer 11 (target) has no Jacobian data.
- **Storage:** HDF5 per checkpoint (`stats.h5` + `spectra.h5` + `metadata.json`). All storage paths are None-tolerant — layer 11 entries use `np.nan`/`None` sentinels.
- **Speed:** ~0.5 s/prompt on B300 after JIT compilation (~9 min/checkpoint, ~23 h on 1 GPU, ~3 h on 8 GPUs).
- **Tests:** 71/71 passing, covering the 3 crash bugs that wasted ~10 h across 4 failed job attempts (`.numpy()` on None, `create_dataset(data=None)`, `str / str` TypeError) plus 4 CLI option bugs.

---

## What's Next

### Done
- ✅ Full 153-checkpoint sweep (spectra, coherence, power-law, spikes, FCR)
- ✅ WMT14 fr-en translation evaluation on all 153 checkpoints
- ✅ Cross-prompt eigenvector overlap on 27 checkpoints (Grassmann overlap)
- ✅ Fluctuation-to-coherent ratio (FCR) computation

### Remaining
1. **CKA block structure across (layer, step):** the signature workspace diagnostic from the original paper — compute CKA(J̄_i, J̄_j) for all (i,j,layer) and track block emergence
2. **MLP-gain curve per checkpoint:** measure how much MLP layers amplify J-lens directions relative to attention layers
3. **Spike-overlap analysis (Direction I #2):** overlap matrix between top eigenvectors of J̄ᵀJ̄ and interpretable concept directions
4. **Three-spectrum NFA test (Direction VI):** weight Gram ≈ AGOP ≈ lens-spike + fluctuation-bulk
5. **Full 153-checkpoint eval sweep:** evaluate all Pythia checkpoints (not just the 27 EleutherAI pre-computed ones)

### Moved to separate project
- **Phase 1 geometric identification** (modular arithmetic, linear probes) — now a standalone project

---

## Files

- **Metric dictionary:** `docs/jlens_metrics.tex` — full LaTeX document with mathematical definitions, motivations, source code index, figure index, data pipeline, bibliography
- **Sweep plan:** `plan.md`
- **Geometric identification plan:** `plan-geom-id.md`
- **Results:** `results/step*/` (153 checkpoints, ~10.4 GB)
- **Figures:** `figures/` (11 publication-quality plots)
- **Eval data:** `eval_data/` (downloaded/generated, gitignored)
- **Tests:** `tests/` (49 tests, 5 files)
- **Scripts:** `scripts/` (sweep, plotting, eval download, prompt, etc.)


## Pythia-12B raw spectrum visualization

An interactive raw-spectrum dashboard now covers every completed common
Pythia-12B checkpoint (wikitext and random-token prompt averages) and every
layer. It animates a side-by-side, log-scale rank spectrum over checkpoints,
with layer coloring and depth-band filtering. The browser view decimates ranks
only; plotted values remain the stored eigenvalues of `Jbar_l^T Jbar_l`.


The companion histogram dashboard, `spectra_histogram_dashboard.html`,
superposes a colored, log-binned eigenvalue-density curve for every layer at
each checkpoint. It deliberately contains no reconstructed Marchenko-Pastur
bulk, so its empirical spectra can be compared directly to that distribution.


For stable playback, histogram spectra are normalized as
`lambda / mean(lambda)` before density estimation on a fixed shared grid. Both dashboards now fix
their ranges from the full checkpoint sweep and update explicit layer-trace
positions without inter-frame interpolation, eliminating frame clipping and
cross-layer transition artifacts.


The raw-rank and histogram dashboards are also emitted as separate Wikitext
and random-token files, avoiding visual overlap between corpus conditions.


The four corpus-specific dashboards use identical x-coordinates in every
checkpoint frame. Playback no longer uses Plotly frame transitions (which tween in
pixel space and kinked); `scripts/dashboard_playback.py` embeds log10(y) per checkpoint
and a JS loop blends y vertically in log space on the fixed x-grid. Histogram bins are fixed across frames; a small
display floor keeps curves continuous on the logarithmic density axis.


Rank-spectrum playback now preserves the full model-rank grid at every
checkpoint. Non-positive or non-finite numerical eigenvalues are displayed at
the smallest positive eigenvalue instead of being dropped, preventing sampled
rank coordinates from changing between animation frames.

### Histogram dashboards: median normalization, linear bins

The histogram dashboards now normalize each spectrum by its **median** (the mean was dominated by
outliers up to ~1e4x the bulk, so the bulk drifted between layers/checkpoints) and plot a
linear-bin density on `[0, 4]` x median with a linear y-axis. Empty bins are exactly zero (no log
display floor, which had created layered 1e-8..1e-4 tails). Eigenvalues beyond 4x median are
outside the window (66-100% of the mass stays inside). Playback blends y linearly for these
(`write_dashboard(..., log_space=False)`).

### Histogram dashboards: normalization removed

Per request, histograms use raw eigenvalues (no mean/median scaling) so layers and checkpoints can
be compared directly. Shared log bins on `[1e-6, 1e3]` (90 bins), y = probability density per
decade on a linear axis, empty bins exactly zero, y-range fixed from the global max peak.
Eigenvalues outside the window are not shown.

### Histogram dashboards: unsmoothed by default

Histogram densities are now raw (no Gaussian smoothing, HISTOGRAM_SMOOTHING_BINS = 0). Smoothing narrowed-peak heights by ~35% at step 512 (peak 5.0 -> 3.2 density per decade); pass --smoothing W to plot_spectrum_histogram.py to re-enable it.

### K2-Horizon-32B spectra computed from fitted lens.pt checkpoints

Hannah's K2-Horizon-32B J-lens fits (`/data/llm_games/hannah/jacobian_lens/out/k2-horizon-32b/stage{1..7}/step*/lens.pt`,
23 checkpoints across 7 training stages, wikitext prompts) only stored the fitted J_l matrices
(jbar), not spectra.h5. Added a one-off script (run via sbatch, never on the login node: holding
all 63 layers' Gram matrices in memory at once peaks at ~25GB RSS) that computes J_l^T J_l and the
eigenvalue spectrum per layer exactly as jlens.spectra.analyze_checkpoint does, one layer at a time
to keep memory low, and writes spectra.h5 next to each lens.pt (same schema as
jlens.storage.save_checkpoint_result) so existing plotting scripts work unmodified. Writes are
atomic (spectra.h5.tmp + rename) so a killed/timed-out job can be resumed by skipping checkpoints
whose spectra.h5 already exists.

### MP fit quality overlay plots (first/last checkpoint, all layers)

Added `scripts/plot_mp_fit_overlay.py`: for a results dir, grids all layers' empirical eigenvalue
histograms against their least-squares Marchenko-Pastur fit (reusing `fit_mp`/`mp_bin_masses` from
plot_mp_fit_r2.py) for the first and last completed checkpoint. Uses matplotlib, not plotly/kaleido:
kaleido's bundled headless Chrome fails to start on this cluster (missing libglib-2.0 system
library), so static PNG export must go through matplotlib's Agg backend instead.

### K2-Horizon-32B histogram dashboard

`scripts/plot_spectrum_histogram.py` assumed a flat `step*` checkpoint layout, which does not fit
K2-Horizon-32B's nested stage{1..7}/step{N} layout (training step numbers collide across stages,
e.g. step500 exists under both stage4 and stage7). Rather than duplicating the script,
`plot_spectra_dashboard.py` gained `discover_checkpoints()`/`checkpoint_label()`, which auto-detect
flat vs. nested layouts and key checkpoints by (stage, step) (stage=0 for flat layouts, so
behavior and output for Pythia-12B are unchanged). `plot_spectrum_histogram.py` now uses these
instead of `spectrum_steps()` directly and takes a `--model-label` flag (default "Pythia-12B") for
the title. Re-run for both models from the same script:
`--results-dir /data/jlens/pythia-12b-deduped ...` and
`--results-dir /data/llm_games/hannah/jacobian_lens/out/k2-horizon-32b --model-label K2-Horizon-32B ...`,
producing figures/pythia-12b/spectra_histogram_wikitext.html (48 checkpoints) and
figures/k2-horizon-32b/spectra_histogram_wikitext.html (23 checkpoints x 63 layers).
`dashboard_playback.write_dashboard`'s slider-label formatting assumed integer frame names;
patched it to fall back to the raw label string for non-integer frame names (e.g. "stage1
step100000"); confirmed Pythia-12B's dashboard still regenerates with comma-formatted integer
labels.

### Continuous colorbar replaces per-layer legend in all spectra dashboards

The per-layer legend (one entry per layer, e.g. "L0", "L1", ... "L35") was unreadable once
there were dozens of layers. Added `layer_colorscale()`/`layer_colorbar_trace()` to
`plot_spectra_dashboard.py` (the shared module) and wired them into all four dashboard
scripts: `plot_spectrum_histogram.py`, `plot_spectra_histogram_dashboard.py`,
`plot_spectrum_dashboard.py`, and `plot_spectra_dashboard.py`'s own two-column
`make_dashboard()`. Each `layer_trace()` now sets `showlegend=False` (dropping `name=`/
`legendgroup=`), and a single invisible zero-size marker trace renders a continuous colorbar
(HSL gradient matching the existing per-layer colors, with ~6 tick labels like L0/L7/.../L35)
instead. The colorbar trace uses a real, finite proxy point rather than `None`/NaN, because
`dashboard_playback.write_dashboard`'s cross-frame x-equality check (`np.array_equal`) fails
on NaN comparisons; it is added once to the figure and once to every frame's data list (two-
column dashboards get exactly one colorbar trace, not one per column).

Regenerated all 7 `figures/*/spectra_*.html` files (5 Pythia-12B single/combined dashboards
rebuilt via `sbatch` on a compute node, not the login node; 2 histogram_wikitext dashboards
for Pythia-12B and K2-Horizon-32B). Verified across all of them: 0 remaining `"name":"L#"`
legend entries, 1 colorbar trace each. Also fixed a recurring relative-output-path bug: the
two-column scripts' `--output` defaults are relative to the script's own working directory,
so running them from `scripts/` without an explicit `--output` writes into
`scripts/figures/pythia-12b/` instead of the repo-root `figures/pythia-12b/`; moved the
mis-written files to the correct location after the job completed.

### Draft paper: free-probability transition in J-lens spectra (MP → semicircle → log-normal)

Compared three two-parameter models (Marchenko-Pastur, log-normal, semicircle-in-log-eigenvalue)
against every (layer, checkpoint) J-lens spectrum for Pythia-12B-deduped (35 layers x 48 steps,
1,680 cells) and K2-Horizon-32B (63 layers x 23 checkpoints, 1,449 cells), fit locally (not on
the cluster) from the shared spectra archive. Found a sharp, theoretically-predicted regime
transition for Pythia-12B-deduped: semicircle-in-log(lambda) is essentially exact at
initialization (mean R^2=0.997, steps <=512, consistent with Bar J_l being a product of
~(L-l) asymptotically-free random matrices at init and the free central limit theorem predicting
a semicircle rather than Gaussian for log(lambda)); it collapses abruptly once training begins
(0% of cells after step 2,000), and log-normal becomes the dominant model (mean R^2=0.978,
62% of all cells), while deep layers (L>=28 of 35) stay MP-best at every step (R^2=0.993, few
multiplicative factors remain near the output). K2-Horizon-32B, which has no near-initialization
checkpoint available, shows log-normal winning 100% of its 1,449 cells, as predicted. Wrote up
as `docs/paper/free_probability_transition.tex` (compiles with pdflatex, 7 pages), with figures
in `docs/paper/figures/` and the underlying per-cell R^2 CSVs
(`docs/paper/model_comparison_grid_{pythia12b,k2}.csv`).

### OLMo-3-1125-32B spectra in normalized units (λ → λ/d_model)

The OLMo-3 figures in `figures/olmo3/` (three-component mixture overlays, composition/R²
heatmap, wikitext histogram dashboard) use eigenvalues divided by d_model = 5120. A first
attempt divided the eigenvalues but kept the raw fit/histogram window [1e-6, 1e3]; for
~26% of layer spectra more than 10% of the mass (up to 78%) then fell below the window, so the
fits saw truncated histograms and the dashboard x-axis no longer matched its bins. Fixed by
moving every window into normalized units ([1e-6, 1e3] / 5120, defined once as `D_MODEL` /
`BINS` in `olmo3_analysis/plot_logskewnormal_fits.py`), so normalization is a pure change of
units. Checked on 4 real steps x 63 layers against the raw-unit `scripts/` fit: every
log-location shifts by a median of exactly ln 5120 = 8.541, median |ΔR²| = 6e-6.

Caveat (independent of normalization): the three-component mixture is weakly identified.
In 46/252 of those fits the MP/semicircle/log-normal weights differ by > 0.05 (up to 0.32)
between the two unit systems at practically equal R² (≤ 3e-3 apart), i.e. the optimizer
lands in different, equally good decompositions. Read per-cell composition in the heatmap
with that in mind; R² is robust. Tests: `tests/test_olmo3_normalization.py`.

### OLMo-3 mixture fit: robust two-pass fitting

Layer 62 at steps 370k/380k fitted at R² 0.935/0.930 although the spectra are identical in
shape to their neighbours (R² 0.997). Cause: optimizer trap. These top-layer spectra are
extremely narrow (IQR ≈ 2.5 fit bins, half the mass in 3 bins); all 4 starts reused the
single-component MP fit's q and only varied the weights, and all ended with MP q on its 1e-3
lower bound. The same cell also fails in raw units, so it is not a normalization artefact.
Fix (`olmo3_analysis/fit_pythia_three_component_mixture.py`): pass 1 adds MP starts
q ∈ {0.01, 0.1, 0.5} (16 starts); pass 2 (`--seed-dir`) restarts each (step, layer) from the
pass-1 optima of the same layer at that step and its two neighbouring steps and keeps the best,
so it can never be worse than pass 1. Over 3843 cells, vs. the single-pass 4-start fit: 0 cells
worse, 2056 better, min R² 0.930 → 0.979, cells with R² < 0.99 851 → 783, layer 62 at
370k/380k → 0.998/0.999; step-to-step composition change (total variation of the 3 weights)
median 0.032 → 0.014, mean 0.066 → 0.045. 36 final fits have MP q at its bound, all in layers
58–62 with R² ≥ 0.9945 (MP acting as a narrow spike for the central peak), so not traps.

### Circle probe on Qwen3.6-27B: cyclic concepts are ordered but open (2026-10-28)

Follow-up to the PH token-cloud cycles (PROJECT1.md), which turned out to be
semantic clusters rather than cyclic orderings. `jlens/circle_probe.py` runs
templated sentences for days/months/seasons/directions/hours/colors plus two
controls (numbers in counting contexts, arbitrary nouns), and captures every
layer's residual (raw and normed), attention value heads, gated attention head
outputs (pre o_proj) and Gated DeltaNet head outputs (pre out_proj): 2,881
units per concept. Template-demeaned word centroids are scored with label
permutations. PCA angle order and LOO cos/sin fits are high almost everywhere,
including the numbers control (an open, curved path gives a PCA horseshoe), so
they test ordinality, not closure. Wrap-aware closure (`circ_minus_lin`,
`closure`) finds loop units at the false-positive rate of the nouns control
(about 1%) for every concept, depth and head type; the median geometry
prefers an open path. Single lead: layer-55 attention heads h4/h0 close
one..twelve in clock contexts but not in counting contexts (not yet
replicated). Tests: `tests/test_circle_probe.py`; job:
`scripts/sweep_circle_probe.sh` (extraction ~4 min on 1 B300, scoring ~1 min on
32 CPUs); figures: `figures/circle_probe_*.png`.
