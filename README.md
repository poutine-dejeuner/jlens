# jlens — Jacobian Lens Spectral Analysis

Study of the spectral evolution of Jacobian lens statistics across Pythia-160M
pretraining checkpoints, based on the [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
framework for detecting emergent global workspaces in transformers.

## Overview

The Jacobian lens computes corpus-averaged Jacobians
$\bar{J}\_\ell = \mathbb{E}_x\!\left[\frac{\partial h_{\text{final}}}{\partial h_\ell}\right]$
measuring how hidden states at layer $\ell$ are transported to the final layer.

This project:
- Computes $\bar{J}\_\ell$ for **153 pretraining checkpoints** of Pythia-160M-deduped
  (steps 1–143,000) on 1,000 WikiText prompts.
- Extracts **spectral diagnostics** (coherence $\kappa$, identity coefficient $a$,
  effective rank, power-law exponent $\alpha$, spike count) per layer.
- Tracks the **four-phase evolution** of the Jacobian workspace: identity →
  collapse → structuration → late plateau.
- Cross-correlates J-lens metrics with **downstream benchmark scores**
  (ARC-Easy, PIQA, SciQ, LAMBADA, etc.) from EleutherAI's pre-computed evals.

**Key finding:** The identity coefficient $a_{\ell=0}$ (how close $\bar{J}$ is to $I$)
has Spearman $\rho = -0.90$ with ARC-Easy accuracy. The coherence $\kappa$
and power-law exponent $\alpha$ are also strongly predictive ($|\rho| > 0.80$).

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) for package management

## Quick Start

```bash
# Install dependencies
uv sync

# Run the pipeline on a single checkpoint (step 1000, 10 prompts, quick test)
uv run jlens run --checkpoints 1000 --n-prompts 10 --dtype float16

# Run on a range of checkpoints (steps 1000-5000, 100 prompts each)
uv run jlens run --n-prompts 100 --start-step 1000 --end-step 5000 --dtype float16

# Analyze completed results
uv run jlens analyze results
```

## Project Structure

```
jlens/
├── jlens/                   # Core Python package
│   ├── _bridge.py           # Imports upstream anthropics/jacobian-lens
│   ├── accumulator.py       # Online Welford accumulation of J̄, J̄ᵀJ̄, κ, a, tr(JᵀJ)
│   ├── checkpoint_manager.py # Pythia checkpoint discovery and loading
│   ├── cli.py               # Click CLI (jlens run, jlens analyze, jlens version)
│   ├── config.py            # PipelineConfig dataclass
│   ├── jacobian.py          # Jacobian computation (wraps upstream jlens.fitting)
│   ├── plotting.py          # Basic plotting utilities
│   ├── prompts.py           # WikiText prompt loading and tokenization
│   ├── runner.py            # Pipeline orchestrator: load → compute → accumulate → save
│   ├── spectra.py           # Spectral analysis: MP fit, power-law tail, spikes
│   └── storage.py           # HDF5 read/write with None-tolerant layer handling
├── scripts/                 # Analysis and plotting scripts
│   ├── prepare_eval_data.py # Download evals + cross-join with J-lens metrics
│   ├── plot_evolution.py    # Spectral evolution plots (6 figures)
│   ├── plot_benchmark_corr.py # J-lens vs benchmark correlation plots (4 figures)
│   ├── plot_eigendist.py    # Eigenvalue distribution grid plot
│   ├── run_slice.py         # SLURM array task runner with resume support
│   ├── sweep_upstream.sh    # SLURM array job for full 153-checkpoint sweep
│   ├── download_evals.py    # Download EleutherAI pre-computed eval JSONs
│   └── prompt_pythia.py     # vLLM-based Pythia prompting client
├── tests/                   # Test suite (49 tests, all passing)
│   ├── test_accumulator.py  # Accumulator edge cases (None tolerance, mixed layers)
│   ├── test_cli.py          # CLI argument parsing (rejects --model, --steps)
│   ├── test_pipeline.py     # Integration tests (full 12-layer Pythia pattern)
│   ├── test_spectra.py      # Spectral analysis (eigen_decompose, MP fit, α)
│   └── test_storage.py      # HDF5 round-trip with missing layers, Path coercion
├── docs/
│   └── jlens_metrics.tex    # Metric dictionary (LaTeX) — definitions, sources, correlations
├── plan.md                  # Direction IV implementation plan
├── plan-geom-id.md          # Algebraic computation subspace identification plan
└── pyproject.toml
```

## Metrics Dictionary

| Metric | Symbol | Definition | Source file |
|--------|--------|------------|-------------|
| Identity coefficient | $a_\ell$ | $\operatorname{tr}(\bar{J}_\ell)/d$ | `accumulator.py:get_a_coeff()` |
| Transport coherence | $\kappa_\ell$ | $\|\bar{J}_\ell\|_F^2 / \mathbb{E}[\|J_\ell\|_F^2]$ | `accumulator.py:get_coherence()` |
| Off-identity norm | $\|R_\ell\|$ | $\|\bar{J}_\ell - a_\ell I\|_F / \sqrt{d}$ | `accumulator.py:get_r_norm()` |
| Transport energy | — | $\mathbb{E}[\operatorname{tr}(J_\ell^\top J_\ell)]$ | `accumulator.py:get_tr_jtj_mean()` |
| Effective rank | $\text{eff\_rank}_\ell$ | $\exp(-\sum p_k \ln p_k)$ | `spectra.py:effective_rank()` |
| Power-law exponent | $\alpha_\ell$ | $\text{CDF}(\lambda) \propto \lambda^\alpha$ | `spectra.py:fit_power_law_tail()` |
| Spike count | $n_{\text{spikes}}$ | $\#\{\lambda_k > 3\lambda_+\}$ | `spectra.py:fit_mp_bulk()` |
| MP $q_{\text{eff}}$ | — | Bulk aspect ratio | `spectra.py:fit_mp_bulk()` |

Full definitions, mathematical derivations, benchmark descriptions, and correlation
tables are in [`docs/jlens_metrics.tex`](docs/jlens_metrics.tex).

## Usage Modes

### Local development / quick tests

```bash
uv run jlens run --checkpoints 1000 --n-prompts 10 --dtype float16
```

### SLURM array sweep (production)

```bash
# Full 153-checkpoint sweep on 8 GPUs
sbatch scripts/sweep_upstream.sh

# Catch up missing checkpoints
python scripts/run_slice.py --model-id EleutherAI/pythia-160m-deduped \
    --checkpoints 8000,9000,10000 --dtype float16
```

### Plotting

```bash
# Spectral evolution across training
uv run python scripts/plot_evolution.py --results-dir results --output-dir figures

# Eigenvalue distribution grid (all checkpoints, all layers)
uv run python scripts/plot_eigendist.py --results-dir results

# J-lens vs benchmark correlations (requires eval data)
uv run python scripts/prepare_eval_data.py   # Step 1: download + cross-join
uv run python scripts/plot_benchmark_corr.py  # Step 2: generate figures
```

### Running tests

```bash
uv run pytest tests/ -v
```

## Training Phase Evolution

| Phase | Steps | $\kappa$ (L0) | $a$ (L0) | Eff. rank | Spikes | Description |
|-------|-------|---------------|-----------|-----------|--------|-------------|
| Identity | 1–64 | 0.94 | 1.00 | 242 | 143 | $\bar{J} \approx I$, near-initialization |
| Collapse | 128–512 | 0.89 → 0.86 | 0.84 → 0.62 | 163 | 101 | Identity breaks, $a$ drops steeply |
| Structuration | 1K–8K | 0.82 → 0.69 | 0.42 → 0.31 | 209 → 241 | 4 | Spikes collapse, power-law bulk forms |
| Late plateau | 16K–143K | 0.53 → 0.37 | 0.21 → 0.17 | 204 → 125 | 1 | Slow convergence, eff. rank drains |

## Benchmark Correlations (Spearman $\rho$)

Top 5 J-lens metrics × benchmarks ($|\rho| > 0.88$, all $p < 0.01$):

| Metric | Benchmark | $\rho$ |
|--------|-----------|--------|
| $a_\ell$ (L0) | ARC-Easy | **−0.901** |
| $n_{\text{spikes}}$ (L0) | SciQ | **−0.894** |
| $n_{\text{spikes}}$ (L0) | ARC-Easy | **−0.885** |
| $\bar{a}$ (mean) | ARC-Easy | **−0.881** |
| $\mathbb{E}[\operatorname{tr}(J^\top J)]$ (L0) | SciQ | **−0.879** |

## References

- **Jacobian Lens** — [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- **Pythia** — Biderman et al., "[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)", ICML 2023
- **Evals** — [EleutherAI/pythia/evals](https://github.com/EleutherAI/pythia/tree/main/evals/pythia-v1/pythia-160m-deduped)
- **LM Evaluation Harness** — [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Tuned Lens** — Belrose et al., "[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)", 2023
