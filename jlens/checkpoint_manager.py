"""Pythia checkpoint discovery and loading.

Checkpoints are stored as HuggingFace branches:
  step1, step2, step4, step8, step16, step32, step64, step128, step256, step512,
  step1000, step2000, ..., step143000
"""

import gc
import logging
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# All Pythia checkpoint steps
_LOG_SPACED = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_EVENLY_SPACED = list(range(1000, 144000, 1000))  # step1000..step143000


def get_checkpoint_steps(start: int = 1, end: int = 143000) -> list[int]:
    """Return sorted list of checkpoint steps in [start, end]."""
    all_steps = sorted(_LOG_SPACED + _EVENLY_SPACED)
    start = int(start)
    end = int(end)
    # step0 exists but skip by default (random init)
    return [s for s in all_steps if start <= s <= end]


def load_checkpoint(
    model_id: str,
    step: int,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Path | None = None,
    device_map: str = "auto",
) -> tuple:
    """Load a Pythia model checkpoint.

    Args:
        model_id: HuggingFace model ID
        step: Training step (branch name)
        dtype: Model dtype
        cache_dir: Cache directory for model weights
        device_map: Device map for model parallelism

    Returns:
        (model, tokenizer) tuple
    """
    revision = f"step{step}"
    logger.info(f"Loading {model_id} @ {revision}")

    kwargs = dict(trust_remote_code=True)
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, **kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map=device_map,
        **kwargs,
    )
    model.eval()

    return model, tokenizer


def unload_checkpoint(model, tokenizer) -> None:
    """Free memory from a loaded checkpoint."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
