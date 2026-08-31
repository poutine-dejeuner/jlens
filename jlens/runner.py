"""Checkpoint iteration runner with resume support."""

import logging
import os
import time
from pathlib import Path

# Disable Triton JIT compilation (B300 compute nodes lack C compiler)
os.environ.setdefault("TRITON_BUILD_WITHOUT_CC", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from tqdm import tqdm

# Set triton build knob before any torch operations
try:
    import triton
    if hasattr(triton.knobs, "build"):
        triton.knobs.build.impl = "dummy"
except ImportError:
    pass

from .accumulator import JacobianAccumulator
from .checkpoint_manager import (
    get_checkpoint_steps,
    load_checkpoint,
    unload_checkpoint,
)
from .config import PipelineConfig
from .jacobian import compute_layer_jacobians_blockwise
from .prompts import load_prompts
from .spectra import analyze_checkpoint
from .storage import is_checkpoint_done, save_checkpoint_result

logger = logging.getLogger(__name__)


def run_checkpoint(
    config: PipelineConfig,
    step: int,
    prompts: list[dict],
    accumulator_class: type = JacobianAccumulator,
) -> dict:
    """Process a single checkpoint.

    Args:
        config: Pipeline configuration
        step: Training step to process
        prompts: Pre-loaded tokenized prompts
        accumulator_class: Accumulator class to use

    Returns:
        Dict with 'stats' and 'spectral' results.
    """
    logger.info(f"=== Processing step {step} ===")

    # Load model
    dtype = getattr(torch, config.model_dtype)
    model, tokenizer = load_checkpoint(
        config.model_id,
        step,
        dtype=dtype,
        cache_dir=config.cache_dir,
    )

    try:
        n_layers = _get_n_layers(model)
        d_model = _get_d_model(model)
        layers = config.layers or list(range(n_layers))

        logger.info(f"Model: {n_layers} layers, d_model={d_model}")
        logger.info(f"Processing layers: {layers}")

        # Initialize accumulator
        accum = accumulator_class(n_layers, d_model, dtype=torch.float32)

        # Process prompts
        for prompt in tqdm(prompts, desc=f"step{step} prompts"):
            input_ids = prompt["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = prompt["attention_mask"].unsqueeze(0).to(model.device)

            jacobians = compute_layer_jacobians_blockwise(
                model, input_ids, attention_mask, layers
            )

            accum.update(jacobians)

        # Get accumulated stats
        stats = accum.get_all_stats()

        # Spectral analysis
        spectral = analyze_checkpoint(stats)

        # Save
        save_checkpoint_result(config.results_dir, step, stats, spectral)

        # Cleanup
        unload_checkpoint(model, tokenizer)

        return {"stats": stats, "spectral": spectral}

    except Exception as e:
        logger.error(f"Error processing step {step}: {e}")
        unload_checkpoint(model, tokenizer)
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

    # Pre-load prompts (same for all checkpoints)
    logger.info("Loading prompts...")
    # Need a tokenizer just for tokenizing prompts
    # Use a cheap temporary model load for tokenization
    _, tokenizer = load_checkpoint(
        config.model_id,
        steps[0],
        dtype=getattr(torch, config.model_dtype),
        cache_dir=config.cache_dir,
    )
    prompts = load_prompts(
        tokenizer,
        n_prompts=config.n_prompts,
        max_seq_len=config.max_seq_len,
        dataset_name=config.prompt_dataset,
        dataset_config=config.prompt_config,
        dataset_split=config.prompt_split,
        seed=config.prompt_seed,
    )
    del tokenizer
    torch.cuda.empty_cache()

    # Process each checkpoint
    errors = []
    t_start = time.time()

    for i, step in enumerate(todo):
        try:
            t0 = time.time()
            run_checkpoint(config, step, prompts)
            elapsed = time.time() - t0
            logger.info(
                f"Step {step} done in {elapsed:.0f}s "
                f"({i+1}/{len(todo)}, {time.time() - t_start:.0f}s total)"
            )
        except Exception as e:
            logger.error(f"Step {step} FAILED: {e}")
            errors.append((step, str(e)))
            # Continue to next checkpoint
            torch.cuda.empty_cache()

    # Summary
    total_time = time.time() - t_start
    logger.info(f"Pipeline complete. {len(todo) - len(errors)}/{len(todo)} succeeded.")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    if errors:
        logger.warning(f"Errors ({len(errors)}):")
        for step, err in errors:
            logger.warning(f"  step {step}: {err}")


def _get_n_layers(model) -> int:
    """Get number of transformer layers from model."""
    if hasattr(model, "gpt_neox"):
        return len(model.gpt_neox.layers)
    elif hasattr(model.model, "layers"):
        return len(model.model.layers)
    else:
        # Fallback: count from config
        if hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        if hasattr(model.config, "n_layer"):
            return model.config.n_layer
        raise ValueError("Cannot determine number of layers")


def _get_d_model(model) -> int:
    """Get model dimension."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model.config, "d_model"):
        return model.config.d_model
    raise ValueError("Cannot determine d_model")
