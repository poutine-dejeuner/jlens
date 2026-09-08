---
name: run-pipeline
description: |
  Run the jlens Jacobian-lens spectral analysis pipeline on Pythia-160M
  checkpoints.  Covers checkpoint discovery, Slurm job submission, resume
  logic, config, and HDF5 result storage.  Use when asked to run jacobians,
  process checkpoints, submit Slurm jobs, or debug pipeline failures.
---

# Run Pipeline — jlens

Run the Jacobian-lens spectral analysis across Pythia-160M pretraining
checkpoints.

## Quick reference

| Task | Command / File |
|---|---|
| List available checkpoints | `uv run python -c "from jlens.checkpoint_manager import get_checkpoint_steps; print(get_checkpoint_steps(1, 143000))"` |
| Full array job (153 checkpoints, 8 GPUs) | `sbatch scripts/run_full.sh` |
| Single checkpoint (interactive) | `uv run python -m jlens.cli run --checkpoints 143000 --n-prompts 100` |
| Resume (skip completed) | Same command — `is_checkpoint_done` skips existing `stepXXXX/` dirs |
| Check what's done | `ls -d results/step*/` |

## What the pipeline does

For each checkpoint step:
1. Load Pythia-160M at that training step in fp16 on GPU.
2. Wrap as upstream `LensModel` via `jlens.from_hf`.
3. For each of N prompts, compute per-layer Jacobians ∂h_final/∂h_ℓ
   using the upstream `jacobian_for_prompt` estimator.
4. Accumulate J̄ (mean) and Gram matrices J̄⊤J̄ online.
5. Run spectral analysis: eigendecomposition, MP bulk fit, power-law
   tail fit, spike count.
6. Save to `results/stepXXXX/`: `stats.h5`, `spectra.h5`, `metadata.json`.

## Slurm submission

Always follow `docs/SLURM.md` rules: explicit `--gres`, `--cpus-per-task`, `--mem`,
`--time`.  Max 1 GPU per job unless multi-GPU code (DDP/FSDP).

**Full run (153 checkpoints × 1000 prompts):**
```bash
sbatch scripts/run_full.sh
```
This splits across 8 array tasks, each processing ~19 checkpoints.

**Single checkpoint test:**
```bash
sbatch scripts/run_single.sh
```

**Resume missing (after partial completion):**
```bash
sbatch scripts/run_missing.sh
```

Check queue: `ssh cursor squeue -u $USER`

## CLI options

```
uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-160m-deduped" \
    --dtype float16 \              # float16, bfloat16, or float32
    --n-prompts 1000 \             # prompts per checkpoint
    --max-seq-len 128 \            # token sequence length
    --prompt-seed 42 \             # seed for prompt shuffling
    --dim-batch 8 \                # output dims per backward pass (VRAM tradeoff)
    --results-dir results \        # output directory
    --checkpoints 143000 \         # comma-separated or use --start/--end
    --layers 0,5,10                # specific layers (or omit for all)
```

## Config

`PipelineConfig` dataclass in `jlens/config.py`.  Key fields:

```
model_id: str = "EleutherAI/pythia-160m-deduped"
model_dtype: str = "float16"
n_prompts: int = 1000
max_seq_len: int = 256
prompt_seed: int = 42
dim_batch: int = 8         # VRAM vs speed tradeoff
checkpoint_step_start: int = 1
checkpoint_step_end: int = 143000
results_dir: Path = Path("results")
```

## Resume logic

`is_checkpoint_done(results_dir, step)` checks for `results/stepXXXX/metadata.json`.
Completed checkpoints are skipped automatically.  To re-run a checkpoint,
delete its `results/stepXXXX/` directory.

## Error recovery

- Crashes are checkpoint-granular: `run_pipeline` catches per-step errors and
  continues to the next step. Failed steps are logged at the end.
- CUDA OOM: reduce `--n-prompts` or `--dim-batch`, or use `--layers` for fewer
  layers per run.
- Triton issues: environment already sets `TRITON_BUILD_WITHOUT_CC=1` and
  `TORCH_COMPILE_DISABLE=1`.
- B300 oddities: use `float16`, not `bfloat16` (RoPE issues on Blackwell).
