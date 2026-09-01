"""Jacobian computation: thin wrapper around anthropics/jacobian-lens.

Uses the upstream `jlens.fitting.jacobian_for_prompt()` estimator — the same
one used in the paper. Takes a `LensModel` (from jlens.from_hf) and returns
per-layer Jacobians in the format our accumulator expects.
"""

import logging
from typing import Optional

import torch

from ._bridge import get_upstream

logger = logging.getLogger(__name__)


def compute_layer_jacobians(
    lens_model,  # jlens.LensModel (from jlens.from_hf)
    text: str,
    layers: list[int],
    *,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
) -> dict[int, torch.Tensor]:
    """Compute per-prompt Jacobian estimator J_l = E[∂h_final/∂h_l].

    Thin wrapper around the upstream estimator from anthropics/jacobian-lens.
    Uses the paper's estimator: one-hot cotangents broadcast over all valid
    target positions, averaged over source positions.

    Args:
        lens_model: LensModel wrapping the HF model (from jlens.from_hf).
        text: Raw prompt text (upstream handles tokenization).
        layers: Layer indices to compute Jacobians for.
        dim_batch: Output dims per backward pass (higher = more VRAM, same FLOPs).
        max_seq_len: Truncate prompt to this many tokens.
        skip_first: Leading token positions to skip (attention sink positions).

    Returns:
        Dict mapping layer -> D×D Jacobian matrix (fp32 CPU).
    """
    upstream = get_upstream()
    jacobians, seq_len, n_valid = upstream.fitting.jacobian_for_prompt(
        lens_model,
        text,
        source_layers=layers,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
    )

    # jacobians is already {layer: tensor[D,D] fp32 cpu} — exactly our format
    return jacobians


def compute_layer_jacobians_blockwise(
    lens_model,
    text: str,
    layers: list[int],
    *,
    dim_batch: int = 8,
    max_seq_len: int = 128,
) -> dict[int, torch.Tensor]:
    """DEPRECATED: kept for backward compatibility.

    Falls through to the upstream estimator.
    """
    logger.warning(
        "compute_layer_jacobians_blockwise is deprecated; "
        "using upstream estimator instead"
    )
    return compute_layer_jacobians(
        lens_model, text, layers,
        dim_batch=dim_batch, max_seq_len=max_seq_len,
    )
