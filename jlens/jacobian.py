"""Jacobian computation: ∂h_final/∂h_ℓ for the Jacobian lens.

Computes per-prompt Jacobians of the final hidden state w.r.t. intermediate
residual stream states. Accumulates J̄ (mean) and Σ (covariance) online.

Key insight: for a transformer with residual connections,
  h_{ℓ+1} = h_ℓ + F_ℓ(h_ℓ)
the final-to-intermediate Jacobian factorizes as a product of per-block
Jacobians M_k = I + ∂F_k/∂h.

We compute J_ℓ(x) = ∂h_final/∂h_ℓ directly via torch.autograd, then
accumulate online using Welford's algorithm for the mean and covariance.
"""

import logging
import os
from typing import Optional

# Disable Triton JIT on B300 nodes
os.environ.setdefault("TRITON_BUILD_WITHOUT_CC", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch

logger = logging.getLogger(__name__)


def compute_layer_jacobians(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Compute ∂h_final/∂h_ℓ for each specified layer.

    For a prompt with seq_len S and d_model D:
      h_ℓ has shape (S, D)
      h_final has shape (S, D)
      J_ℓ = ∂h_final/∂h_ℓ has shape (S, D, D) — one D×D Jacobian per position.

    We average over sequence positions to get a single D×D Jacobian per layer.
    This is valid because the Jacobian is position-independent in theory
    (causal mask means position i only depends on positions ≤ i, but the
    derivative ∂(h_final)_i/∂(h_ℓ)_j is zero for j > i and the non-zero
    block is approximately Toeplitz for long sequences).

    Args:
        model: HuggingFace model
        input_ids: (1, S) token ids
        attention_mask: (1, S) attention mask
        layers: list of layer indices

    Returns:
        Dict mapping layer → D×D Jacobian matrix (averaged over positions)
    """
    S = input_ids.shape[1]

    # Capture intermediate hidden states WITHOUT detaching from the graph.
    # We use forward hooks that store the raw output (still connected to the graph).
    hidden_states: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(idx):
        def hook(module, inputs, output):
            # Store the raw output - do NOT detach, keep it in the graph
            # output is a tuple for GPT-NeoX layers, first element is hidden states
            if isinstance(output, tuple):
                hidden_states[idx] = output[0]
            else:
                hidden_states[idx] = output
        return hook

    # For GPT-NeoX (Pythia), layers are model.gpt_neox.layers[idx]
    if hasattr(model, "gpt_neox"):
        layer_module = model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_module = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers in model")

    for layer_idx in layers:
        h = layer_module[layer_idx].register_forward_hook(_make_hook(layer_idx))
        hooks.append(h)

    try:
        # Forward pass with output_hidden_states to get the final hidden state
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get final hidden state (from model output, still in graph)
        h_final = outputs.hidden_states[-1]  # (1, S, D)

        # Compute full Jacobian per layer
        jacobians = {}
        last_pos = int(attention_mask.sum().item()) - 1
        D = h_final.shape[-1]

        n_total = len(layers) * D
        count = 0

        for layer_idx in layers:
            h_ell = hidden_states[layer_idx]  # (1, S, D)

            # Jacobian: ∂(h_final[last_pos]) / ∂(h_ell[last_pos])
            # We compute row by row using sequential vjp.
            jac = torch.zeros(D, D, device=h_ell.device, dtype=h_ell.dtype)

            for d in range(D):
                # Create grad_output that selects dimension d at the last position
                grad_output = torch.zeros_like(h_final)
                grad_output[0, last_pos, d] = 1.0

                count += 1
                is_last = (count == n_total)

                grads = torch.autograd.grad(
                    outputs=h_final,
                    inputs=h_ell,
                    grad_outputs=grad_output,
                    retain_graph=not is_last,
                    create_graph=False,
                    allow_unused=False,
                )[0]  # (1, S, D)

                # Take the gradient w.r.t. the last position of h_ell
                jac[d] = grads[0, last_pos, :]

            jacobians[layer_idx] = jac

        return jacobians

    finally:
        for h in hooks:
            h.remove()


def compute_layer_jacobians_blockwise(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Compute ∂h_final/∂h_ℓ using per-block Jacobian factorization.

    For a residual transformer: h_{k+1} = h_k + F_k(h_k).

    The per-block Jacobian M_k = ∂h_{k+1}/∂h_k = I + ∂F_k/∂h_k.
    Then: J_ℓ = M_{L-1} · M_{L-2} · ... · M_ℓ.

    Computing M_k only backpropagates through ONE transformer block,
    which is ~14× faster than backpropagating through the full model.

    Args:
        model: HuggingFace model (GPT-NeoX/Pythia)
        input_ids: (1, S) token ids
        attention_mask: (1, S) attention mask
        layers: list of layer indices to compute Jacobians for

    Returns:
        Dict mapping layer → D×D Jacobian matrix at last position
    """
    S = input_ids.shape[1]
    device = input_ids.device
    D = model.config.hidden_size

    if not hasattr(model, "gpt_neox"):
        raise ValueError("Only GPT-NeoX (Pythia) models supported")

    transformer_layers = model.gpt_neox.layers
    L = len(transformer_layers)
    last_pos = int(attention_mask.sum().item()) - 1
    position_ids = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0)

    # Create causal + padding attention mask (4D bool for SDPA)
    causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
    # Expand padding: positions with mask=0 should be masked
    pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S)
    full_mask = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask  # (1, 1, S, S)

    # Step 1: Forward pass (no_grad) to get hidden states at each layer
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hs[0] = embedding output
        # hs[1..L] = transformer block 0..L-1 outputs
        # hs[L] already includes final_layer_norm
        hs = [h.clone() for h in outputs.hidden_states]
    
    n_hs = len(hs)
    assert n_hs == L + 1, f"Expected {L+1} hidden states (embed + {L} blocks), got {n_hs}"

    # Step 2: Per block, compute M_k = ∂(hs[k+1]) / ∂(hs[k])
    # For k = 0..L-2: hs[k+1] = block_k(hs[k])
    # For k = L-1 (last block): hs[L] = final_ln(block_{L-1}(hs[L-1]))
    M_blocks = {}
    rotary_emb = model.gpt_neox.rotary_emb
    final_ln = model.gpt_neox.final_layer_norm

    for k in range(L):
        h_in = hs[k].clone().detach().requires_grad_(True)  # (1, S, D)
        pos_emb = rotary_emb(h_in, position_ids)

        block_out = transformer_layers[k](
            h_in,
            attention_mask=full_mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )
        if isinstance(block_out, tuple):
            block_out = block_out[0]

        # Last block: apply final layer norm
        if k == L - 1:
            h_out = final_ln(block_out)
        else:
            h_out = block_out

        # M_k = ∂(h_out[last_pos]) / ∂(h_in[last_pos])
        M = torch.zeros(D, D, device=device, dtype=h_out.dtype)

        for d in range(D):
            grad_output = torch.zeros_like(h_out)
            grad_output[0, last_pos, d] = 1.0

            grads = torch.autograd.grad(
                outputs=h_out,
                inputs=h_in,
                grad_outputs=grad_output,
                retain_graph=(d < D - 1),
                create_graph=False,
                allow_unused=False,
            )[0]

            M[d] = grads[0, last_pos, :]

        # Free graph for this block
        h_out = None
        h_in = None
        grads = None

        M_blocks[k] = M

    # Step 3: Compute J_ℓ = M_{L-1} · ... · M_ℓ
    jacobians = {}
    for ell in layers:
        J = torch.eye(D, device=device, dtype=M_blocks[ell].dtype)
        for k in range(ell, L):
            J = M_blocks[k] @ J
        jacobians[ell] = J

    return jacobians
