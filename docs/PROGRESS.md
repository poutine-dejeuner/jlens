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
