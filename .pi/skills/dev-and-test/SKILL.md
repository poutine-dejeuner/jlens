---
name: dev-and-test
description: |
  Development workflow for jlens: run tests, fix bugs, understand the
  architecture, and add features.  Covers the test suite, package structure,
  the upstream bridge (anthropics/jacobian-lens), and coding conventions.
  Use when modifying jlens code, debugging failures, or adding functionality.
---

# Dev & Test — jlens

Development and testing workflow for the jlens codebase.

## Test suite

```bash
# All tests (58 tests, ~30s)
PYTHONPATH="$(pwd):$PYTHONPATH" uv run pytest tests/ -v

# Specific test file
PYTHONPATH="$(pwd):$PYTHONPATH" uv run pytest tests/test_storage.py -v

# Single test
PYTHONPATH="$(pwd):$PYTHONPATH" uv run pytest tests/test_storage.py::TestSaveLoadRoundtrip -v

# Skip slow tests (network-dependent prompt loading)
PYTHONPATH="$(pwd):$PYTHONPATH" uv run pytest tests/ -v -k "not TestPromptsSeed"
```

**Always set `PYTHONPATH`** — the package uses implicit relative imports
and is not installed as an editable package.

## Test layout

```
tests/
├── test_accumulator.py   # Online J̄ accumulation, Welford-style stats
├── test_spectra.py       # Eigen decomposition, MP fit, power-law, spike count
├── test_storage.py       # HDF5 save/load round-trip, checkpoint detection
├── test_pipeline.py      # End-to-end: accumulator → spectra → storage
├── test_cli.py           # CLI option parsing, help output
└── test_bugs.py          # Regression tests for fixed bugs
```

## Package architecture

```
jlens/
├── _bridge.py            # sys.path hack: imports upstream anthropics/jacobian-lens
├── accumulator.py        # Online accumulation of J̄, J⊤J Gram, κ stats
├── checkpoint_manager.py # Discover, load, unload Pythia-160M checkpoints
├── cli.py                # Click CLI: run, analyze, version
├── config.py             # PipelineConfig dataclass
├── jacobian.py           # Thin wrapper: upstream jacobian_for_prompt estimator
├── plotting.py           # Heatmap utilities
├── prompts.py            # Wikitext prompt loader with shuffling
├── runner.py             # Checkpoint iteration loop with resume + error recovery
├── spectra.py            # Eigen decomposition, MP bulk fit, power-law tail, spike count
└── storage.py            # HDF5 save/load, _or_nan/_or_nan_int helpers
```

## Key data flow

```
checkpoint → load model → wrap LensModel
    → for each prompt:
        jacobian_for_prompt(lens_model, text, layers)
        → {layer: tensor[d,d] fp32 cpu}
        → accumulator.update(jacobians)
    → stats = accumulator.get_all_stats()
        {n, jbar, jtj_gram, coherence, a_coeff, r_norm, tr_jtj_mean}
    → spectral = analyze_checkpoint(stats)
        {eigenvalues, effective_rank, q_eff, n_spikes, power_law_alpha, mp_sigma2}
    → save_checkpoint_result(results_dir, step, stats, spectral)
```

## The upstream bridge (`_bridge.py`)

`jlens` wraps [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).
The bridge:

1. Removes local `jlens` from `sys.modules`.
2. Removes local paths from `sys.path`.
3. Imports the upstream `jlens` package.
4. Restores local state after import.

This is a fragile hack.  If imports break:
- The upstream package must be pip-installed in the venv.
- Check `uv run pip show jacobian-lens` to verify.
- Never run two threads during the import window.

## Coding conventions

- **Test-based development.**  Every bug gets a regression test before fixing.
- **Use `np.random.default_rng(seed)`** for reproducible test data, not the
  legacy `np.random.seed()`.
- **HDF5 attrs must be scalar types** (int, float, str).  Use `_or_nan_int`
  for integer attrs that may be NaN (empty layers).  It converts NaN → -1.
- **`_or_nan(x)`** for float attrs that may be None or NaN.
- **GPU models use `float16`** on B300 (not `bfloat16` — RoPE issues).

## Common fixes

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: jlens.spectra` | Set `PYTHONPATH=$(pwd)` |
| Triton JIT compilation error | Already set: `TRITON_BUILD_WITHOUT_CC=1` |
| CUDA OOM | Reduce `dim_batch` (default 8 → try 4) or `n_prompts` |
| `KeyError` in compare_estimators | Script uses `f["layer_N"]["jbar"][:]`, not `f[str(N)][:]` |
| Empty layer n_spikes = NaN | Fixed: `_or_nan_int` now converts NaN → -1 |
| `--prompt-seed` has no effect | Fixed: both `load_prompts` functions now shuffle |
