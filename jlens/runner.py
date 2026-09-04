"""Checkpoint iteration runner with resume support."""

import logging
import os
import time
from pathlib import Path

# Disable Triton JIT compilation (B300 compute nodes lack C compiler)
os.environ.setdefault("TRITON_BUILD_WITHOUT_CC", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import numpy as np
from tqdm import tqdm

# Set triton build knob before any torch operations
try:
    import triton
    if hasattr(triton.knobs, "build"):
        triton.knobs.build.impl = "dummy"
except ImportError:
    pass

from ._bridge import get_upstream
from .accumulator import JacobianAccumulator
from .checkpoint_manager import (
    get_checkpoint_steps,
    load_checkpoint,
    unload_checkpoint,
)
from .config import PipelineConfig
from .jacobian import compute_layer_jacobians
from .spectra import analyze_checkpoint, subspace_overlap
from .storage import is_checkpoint_done, save_checkpoint_result

logger = logging.getLogger(__name__)


def _wrap_model(hf_model, tokenizer):
    """Wrap an HF model as a LensModel using upstream jlens.from_hf.

    Auto-detects layout, with a fallback for Pythia-160M which uses
    lm_head instead of embed_out.
    """
    upstream = get_upstream()
    try:
        return upstream.from_hf(hf_model, tokenizer)
    except ValueError as e:
        if not hasattr(hf_model, 'gpt_neox') or not hasattr(hf_model, 'lm_head'):
            raise
        msg = str(e).lower()
        if 'layout' not in msg and 'embed' not in msg and 'lm_head' not in msg:
            raise
        layout = upstream.Layout(
            path='gpt_neox',
            layers='layers',
            norm='final_layer_norm',
            embed='embed_in',
            lm_head='lm_head',
        )
        return upstream.from_hf(hf_model, tokenizer, layout=layout)


def run_checkpoint(
    config: PipelineConfig,
    step: int,
    prompts: list[str],
    accumulator_class: type = JacobianAccumulator,
) -> dict:
    """Process a single checkpoint.

    Args:
        config: Pipeline configuration
        step: Training step to process
        prompts: Raw text prompts (upstream jlens handles tokenization).
        accumulator_class: Accumulator class to use

    Returns:
        Dict with 'stats' and 'spectral' results.
    """
    logger.info(f"=== Processing step {step} ===")

    # Load model
    dtype = getattr(torch, config.model_dtype)
    hf_model, tokenizer = load_checkpoint(
        config.model_id,
        step,
        dtype=dtype,
        cache_dir=config.cache_dir,
    )

    try:
        # Wrap as LensModel using upstream adapter (handles GPT-NeoX/Pythia)
        lens_model = _wrap_model(hf_model, tokenizer)

        n_layers = lens_model.n_layers
        d_model = lens_model.d_model
        layers = config.layers or list(range(n_layers - 1))  # upstream: source < target_layer = n_layers-1

        logger.info(f"Model: {n_layers} layers, d_model={d_model}")
        logger.info(f"Processing layers: {layers}")

        # Initialize accumulator
        accum = accumulator_class(n_layers, d_model, dtype=torch.float32)

        # Process prompts (now raw text, upstream handles tokenization)
        from tqdm import tqdm
        for text in tqdm(prompts, desc=f"step{step} prompts"):
            jacobians = compute_layer_jacobians(
                lens_model,
                text,
                layers,
                dim_batch=config.dim_batch,
                max_seq_len=config.max_seq_len,
            )

            accum.update(jacobians)

        # Get accumulated stats
        stats = accum.get_all_stats()

        # Spectral analysis
        spectral = analyze_checkpoint(stats)

        # Save
        save_checkpoint_result(config.results_dir, step, stats, spectral)

        # Cleanup
        unload_checkpoint(hf_model, tokenizer)
        del lens_model

        return {"stats": stats, "spectral": spectral}

    except Exception as e:
        logger.error(f"Error processing step {step}: {e}")
        unload_checkpoint(hf_model, tokenizer)
        raise


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full checkpoint pipeline with resume support.

    Args:
        config: Pipeline configuration
    """
    config.results_dir = Path(config.results_dir)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoints to process
    if config.checkpoints:
        steps = [int(s) for s in config.checkpoints]
    else:
        steps = get_checkpoint_steps(
            config.checkpoint_step_start,
            config.checkpoint_step_end,
        )

    logger.info(f"Pipeline: {len(steps)} checkpoints to process")
    logger.info(f"Results dir: {config.results_dir}")

    # Filter out already-completed checkpoints
    todo = []
    skipped = 0
    for step in steps:
        if is_checkpoint_done(config.results_dir, step):
            skipped += 1
        else:
            todo.append(step)

    logger.info(f"Already done: {skipped}, Remaining: {len(todo)}")

    if not todo:
        logger.info("Nothing to do!")
        return

    # Pre-load prompts as raw text (upstream jlens handles tokenization per prompt)
    logger.info("Loading prompts...")
    # Use a cheap temporary model load just for the tokenizer
    _, tokenizer = load_checkpoint(
        config.model_id,
        steps[0],
        dtype=getattr(torch, config.model_dtype),
        cache_dir=config.cache_dir,
    )
    from .prompts import load_prompts_text
    prompts = load_prompts_text(
        tokenizer,
        n_prompts=config.n_prompts,
        max_seq_len=config.max_seq_len,
        dataset_name=config.prompt_dataset,
        dataset_config=config.prompt_config,
        dataset_split=config.prompt_split,
        seed=config.prompt_seed,
    )

    # Optionally load overlap prompts (different seed, same dataset)
    overlap_prompts = None
    if getattr(config, 'overlap_seed', None) is not None:
        from .prompts import load_prompts_text as lpt, save_prompts_text
        overlap_n = getattr(config, 'overlap_n_prompts', None) or config.n_prompts
        logger.info(f"Loading overlap prompts (seed={config.overlap_seed}, n={overlap_n})...")
        overlap_prompts = lpt(
            tokenizer,
            n_prompts=overlap_n,
            max_seq_len=config.max_seq_len,
            dataset_name=config.prompt_dataset,
            dataset_config=config.prompt_config,
            dataset_split=config.prompt_split,
            seed=config.overlap_seed,
        )
        save_prompts_text(overlap_prompts, seed=config.overlap_seed)

    del tokenizer
    torch.cuda.empty_cache()

    # Process each checkpoint
    errors = []
    t_start = time.time()

    for i, step in enumerate(todo):
        try:
            t0 = time.time()
            if overlap_prompts is not None:
                run_checkpoint_with_overlap(config, step, prompts, overlap_prompts)
            else:
                run_checkpoint(config, step, prompts)
            elapsed = time.time() - t0
            logger.info(
                f"Step {step} done in {elapsed:.0f}s "
                f"({i+1}/{len(todo)}, {time.time() - t_start:.0f}s total)"
            )
        except Exception as e:
            logger.error(f"Step {step} FAILED: {e}")
            errors.append((step, str(e)))
            torch.cuda.empty_cache()

    # Summary
    total_time = time.time() - t_start
    logger.info(f"Pipeline complete. {len(todo) - len(errors)}/{len(todo)} succeeded.")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    if errors:
        logger.warning(f"Errors ({len(errors)}):")
        for step, err in errors:
            logger.warning(f"  step {step}: {err}")


def run_checkpoint_with_overlap(
    config: PipelineConfig,
    step: int,
    prompts: list[str],
    overlap_prompts: list[str],
    accumulator_class: type = JacobianAccumulator,
) -> dict:
    """Process a single checkpoint with cross-prompt eigenvector overlap.

    Splits prompts into two independent batches:
      batch_A = prompts       (main sweep batch)
      batch_B = overlap_prompts  (held-out batch)

    Computes J̄_A and J̄_B per layer via independent accumulators,
    then measures the Grassmann subspace overlap of their top-k
    eigenvectors of J̄⊤J̄.

    Args:
        config: Pipeline configuration
        step: Training step to process
        prompts: Main batch of raw text prompts
        overlap_prompts: Held-out batch for overlap measurement
        accumulator_class: Accumulator class to use

    Returns:
        Dict with 'stats', 'spectral', and 'overlap' results.
    """
    logger.info(f"=== Processing step {step} (with overlap) ===")

    dtype = getattr(torch, config.model_dtype)
    hf_model, tokenizer = load_checkpoint(
        config.model_id,
        step,
        dtype=dtype,
        cache_dir=config.cache_dir,
    )

    try:
        lens_model = _wrap_model(hf_model, tokenizer)
        n_layers = lens_model.n_layers
        d_model = lens_model.d_model
        layers = config.layers or list(range(n_layers - 1))

        logger.info(f"Model: {n_layers} layers, d_model={d_model}")
        logger.info(f"Main prompts: {len(prompts)}, Overlap prompts: {len(overlap_prompts)}")

        # Two independent accumulators
        accum_main = accumulator_class(n_layers, d_model, dtype=torch.float32)
        accum_overlap = accumulator_class(n_layers, d_model, dtype=torch.float32)

        # Process main prompts
        from tqdm import tqdm
        for text in tqdm(prompts, desc=f"step{step} main"):
            jacobians = compute_layer_jacobians(
                lens_model, text, layers,
                dim_batch=config.dim_batch,
                max_seq_len=config.max_seq_len,
            )
            accum_main.update(jacobians)

        # Process overlap prompts (held-out set)
        for text in tqdm(overlap_prompts, desc=f"step{step} overlap"):
            jacobians = compute_layer_jacobians(
                lens_model, text, layers,
                dim_batch=config.dim_batch,
                max_seq_len=config.max_seq_len,
            )
            accum_overlap.update(jacobians)

        # Compute eigenvector overlap per layer
        overlap_list = []
        for i in range(n_layers):
            gram_a = accum_main.get_gram(i)
            gram_b = accum_overlap.get_gram(i)
            if gram_a is not None and gram_b is not None:
                ov = subspace_overlap(gram_a.numpy(), gram_b.numpy())
            else:
                ov = np.nan
            overlap_list.append(ov)
            logger.info(f"  L{i:02d}: overlap = {ov:.4f}")

        # Main stats from the primary accumulator
        stats = accum_main.get_all_stats()

        # Spectral analysis on main stats (includes sigma, fcr)
        spectral = analyze_checkpoint(stats)
        spectral["eigenvector_overlap"] = overlap_list

        # Save
        save_checkpoint_result(config.results_dir, step, stats, spectral)

        # Cleanup
        unload_checkpoint(hf_model, tokenizer)
        del lens_model

        return {"stats": stats, "spectral": spectral, "overlap": overlap_list}

    except Exception as e:
        logger.error(f"Error processing step {step}: {e}")
        unload_checkpoint(hf_model, tokenizer)
        raise


