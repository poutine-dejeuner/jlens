"""Thought manifold: eigendecomposition of pairwise cosine similarity of
token hidden-state representations.

Two modes (Option A and Option B):

Option A — Singleton (isolated token):
    For each vocabulary token i, run [i] → forward → h_i ∈ R^d.
    Stack X ∈ R^{V×d}, compute Gram+eigh of X̂X̂ᵀ.
    Captures the static vocabulary geometry (language families,
    subword morphology) without any contextual disambiguation.

Option B — Contextualized (in-context average):
    For each vocabulary token, collect N sentences from Wikipedia EN
    where the token appears, run a forward pass on the full sentence,
    extract the hidden state at the token's position, and average.
    Then stack X ∈ R^{V×d} and compute Gram+eigh.
    Captures the contextualized semantics: part-of-speech, animacy,
    concreteness, etc.

Algorithm (shared, Gram+eigh approach for large V):

    1.  Collect per-token representations by either mode.
    2.  L2-normalize rows: X̂ = X / ‖X‖₂ (row-wise).
    3.  Form the d×d Gram matrix:  G = X̂ᵀ X̂.
    4.  Eigendecompose G → (λ_g, V).
    5.  Left eigenvectors: U = X̂ V diag(1/√λ_g).

Why Gram+eigh over thin SVD of X̂:
    - The d×d Gram matrix (5120² × 4 bytes = 105 MB) is much smaller than
      X̂ (248K × 5K × 4 = 5.1 GB).  eigh on 5120×5120 uses ~450M FLOPs.
    - The one large matmul X̂ᵀ X̂ costs O(V d²) ≈ 6.5 TFLOPs — same as SVD.
    - eigh on a small float32 PSD matrix is numerically bulletproof.

References:
    - "Finding structure with randomness" (Halko, Martinsson, Tropp 2011)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ._bridge import get_upstream
from .checkpoint_manager import load_checkpoint, unload_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option A: singleton forward passes
# ---------------------------------------------------------------------------


def compute_token_representations(
    lens_model,
    token_ids: Optional[list[int]] = None,
    batch_size: int = 512,
    max_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Compute final-layer residual-stream representations for vocabulary tokens.

    For each token ID, runs a singleton forward pass (input_ids=[token_id]),
    reads the final-layer hidden state at position 0, and stacks them.

    The hidden state is taken *post* the final RMS norm, i.e. the
    representation that the LM head sees for producing logits.

    Args:
        lens_model: HFLensModel wrapping the model.
        token_ids: Specific token IDs to compute.  If None, uses all tokens
            from 0 to max_token_id (or vocab_size - 1).
        batch_size: Tokens to process per forward pass.  Each forward pass
            runs the model on a batch of single-token sequences.
        max_token_id: Upper bound (exclusive) when token_ids is None.
            Defaults to vocab_size (from config).

    Returns:
        Tensor of shape (V, d_model) in float32 on CPU, where V is the
        number of tokens processed.
    """
    upstream = get_upstream()

    if max_token_id is None:
        hf_config = lens_model._hf_model.config
        if hasattr(hf_config, "text_config"):
            max_token_id = hf_config.text_config.vocab_size
        else:
            max_token_id = hf_config.vocab_size

    if token_ids is None:
        token_ids = list(range(max_token_id))

    n_tokens = len(token_ids)
    d_model = lens_model.d_model
    device = lens_model.input_device

    logger.info(
        "Computing hidden representations for %d tokens (d_model=%d, batch_size=%d)",
        n_tokens,
        d_model,
        batch_size,
    )

    all_hidden = torch.zeros(n_tokens, d_model, dtype=torch.float32)

    n_batches = (n_tokens + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_tokens)
        batch_token_ids = token_ids[start:end]

        # Each input is a single-token sequence: shape (batch, 1)
        input_ids = torch.tensor(
            [[tid] for tid in batch_token_ids],
            dtype=torch.long,
            device=device,
        )

        # Run forward pass.  The text module (for Qwen3_5TextModel) returns:
        #   BaseModelOutputWithPast(last_hidden_state=..., ...)
        # where last_hidden_state has shape (batch, 1, d_model).
        output = lens_model.forward(input_ids)

        # Unwrap: forward() may return a tuple; the first element is
        # last_hidden_state (or a BaseModelOutputWithPast).
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            hidden = output[0]
            if hasattr(hidden, "last_hidden_state"):
                hidden = hidden.last_hidden_state
        else:
            hidden = output

        # hidden shape: (batch, 1, d_model) → squeeze to (batch, d_model)
        hidden = hidden.squeeze(1).float()

        all_hidden[start:end] = hidden.cpu()

        if batch_idx % 50 == 0:
            logger.debug(
                "  batch %d/%d (%d tokens)",
                batch_idx + 1,
                n_batches,
                end,
            )

    logger.info("Token representations: shape=%s, memory=%.1f MB",
                all_hidden.shape, all_hidden.element_size() * all_hidden.numel() / 1e6)

    return all_hidden


# ---------------------------------------------------------------------------
# Option B: contextualized averaging
# ---------------------------------------------------------------------------


def compute_contextual_representations(
    hf_model,
    token_ids: list[int],
    n_contexts: int = 100,
    min_sentence_len: int = 20,
    max_sentence_len: int = 200,
    batch_size: int = 16,
    wikipedia_lang: str = "20231101.en",
) -> torch.Tensor:
    """Compute contextualized token representations by averaging hidden
    states from in-context occurrences in Wikipedia sentences (Option B).

    For each target token, collect up to ``n_contexts`` sentences from
    Wikipedia EN where the token appears.  Each sentence is run through
    the model in a single forward pass, extracting ``last_hidden_state``
    at all positions.  The hidden state at the exact position of the
    target token is accumulated; after the scan per-token averages are
    returned.

    Args:
        hf_model: HuggingFace ``AutoModelForCausalLM`` already on GPU.
        token_ids: Target vocabulary token IDs to collect contexts for.
        n_contexts: Target number of contexts per token (actual may be
            lower for rare tokens).
        min_sentence_len, max_sentence_len: Filter sentences by character
            length before tokenization.
        batch_size: Sentences per forward pass (tuned for B300 48 GB).
        wikipedia_lang: Wikipedia language split to stream.

    Returns:
        Tensor of shape (V, d_model) in float32 on CPU.  Tokens not found
        in the corpus have zero vectors.
    """
    from collections import defaultdict
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Handle both multimodal (text_config) and plain (config) models
    if hasattr(hf_model.config, "text_config"):
        d_model = hf_model.config.text_config.hidden_size
    else:
        d_model = hf_model.config.hidden_size
    device = hf_model.device
    target_set = set(token_ids)

    tok = AutoTokenizer.from_pretrained(
        hf_model.config._name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Detect text module: Qwen3.5 uses .model, GPT2/others use .transformer
    if hasattr(hf_model, "model"):
        text_model = hf_model.model
    elif hasattr(hf_model, "transformer"):
        text_model = hf_model.transformer
    else:
        raise AttributeError(
            f"Cannot find text module on {type(hf_model).__name__}. "
            "Expected .model or .transformer.")

    logger.info(
        "Collecting %d contexts per token for %d target tokens "
        "(d_model=%d, batch_size=%d)",
        n_contexts, len(token_ids), d_model, batch_size,
    )

    sums = defaultdict(lambda: torch.zeros(d_model, dtype=torch.float32))
    counts = defaultdict(int)

    ds = load_dataset(
        "wikimedia/wikipedia", wikipedia_lang, split="train", streaming=True)

    if not target_set:
        logger.info('No target tokens to collect; returning empty tensor.')
        return torch.zeros(len(token_ids), d_model, dtype=torch.float32)

    pending = set(target_set)
    n_sentences = 0
    n_total_contexts = 0

    for article_idx, row in enumerate(ds):
        text = row["text"]
        raw_sents = text.replace("\n", " ").split(". ")
        sentences = [s.strip() for s in raw_sents
                     if min_sentence_len <= len(s.strip()) <= max_sentence_len]

        if article_idx < 5 or article_idx % 100 == 0:
            logger.info(
                "article %d: text_len=%d, %d raw sents, %d qualifying",
                article_idx, len(text), len(raw_sents), len(sentences))

        for sent_batch_start in range(0, len(sentences), batch_size):
            sent_batch = sentences[sent_batch_start:sent_batch_start + batch_size]
            if not sent_batch:
                break

            encoded = tok(sent_batch, padding=True, truncation=True,
                          max_length=max_sentence_len, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            B, L = input_ids.shape

            with torch.no_grad():
                outputs = text_model(input_ids)
                hidden = outputs.last_hidden_state.float()  # (B, L, d)

            # GPU-side matching: find all positions where any target token appears
            target_tensor = torch.tensor(list(target_set), device=device, dtype=torch.long)
            mask = torch.isin(input_ids, target_tensor)  # (B, L) bool

            # For each matching position, accumulate to CPU sums
            flat_mask = mask.reshape(-1)
            if flat_mask.any():
                flat_ids = input_ids.reshape(-1)[flat_mask].cpu()
                flat_hidden = hidden.reshape(-1, d_model)[flat_mask].cpu()

                for tid, vec in zip(flat_ids.tolist(), flat_hidden):
                    if counts[tid] < n_contexts:
                        sums[tid] += vec
                        counts[tid] += 1
                        n_total_contexts += 1
                        if counts[tid] == n_contexts:
                            pending.discard(tid)

            n_sentences += B

        if article_idx % 50 == 0 and len(target_set) > 0:
            pct_done = 100 * (len(target_set) - len(pending)) / len(target_set)
            logger.info(
                "  article %d: %d sentences, %d contexts, "
                "%.1f%% tokens saturated (%d pending)",
                article_idx, n_sentences, n_total_contexts,
                pct_done, len(pending),
            )

        if not pending:
            logger.info(
                "All %d target tokens saturated at article %d (%d sentences)",
                len(target_set), article_idx, n_sentences)
            break

    X = torch.zeros(len(token_ids), d_model, dtype=torch.float32)
    for idx, tid in enumerate(token_ids):
        if counts[tid] > 0:
            X[idx] = sums[tid] / counts[tid]

    n_found = sum(1 for c in counts.values() if c > 0)
    n_saturated = sum(1 for c in counts.values() if c >= n_contexts)
    logger.info(
        "Contextual representations: %d tokens found (%d saturated), "
        "%d contexts total, %.1f avg contexts per token",
        n_found, n_saturated, n_total_contexts,
        n_total_contexts / max(n_found, 1),
    )

    return X


def compute_contextual_fast(
    hf_model,
    token_ids: list[int],
    n_sentences: int = 500_000,
    min_sentence_len: int = 20,
    max_sentence_len: int = 200,
    batch_size: int = 16,
    wikipedia_lang: str = "20231101.en",
) -> torch.Tensor:
    """Compute contextualized token representations via random sentence
    sampling from Wikipedia (Option B - fast).

    Instead of sequentially scanning all of Wikipedia, this samples
    ``n_sentences`` random sentences, tokenizes them in batches, runs
    forward passes, and aggregates hidden states per token position.
    After all sentences are processed, returns per-token averages.

    This is ~100× faster than the full sequential scan and gives
    statistically equivalent results thanks to the law of large numbers.

    Args:
        hf_model: HuggingFace ``AutoModelForCausalLM`` already on GPU.
        token_ids: Target vocabulary token IDs.
        n_sentences: Number of random Wikipedia sentences to process.
        batch_size: Sentences per forward pass.
    """
    from collections import defaultdict
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random

    # Detect model config
    if hasattr(hf_model.config, "text_config"):
        d_model = hf_model.config.text_config.hidden_size
    else:
        d_model = hf_model.config.hidden_size
    device = hf_model.device
    target_set = set(token_ids)

    tok = AutoTokenizer.from_pretrained(
        hf_model.config._name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Detect text module
    if hasattr(hf_model, "model"):
        text_model = hf_model.model
    elif hasattr(hf_model, "transformer"):
        text_model = hf_model.transformer
    else:
        raise AttributeError(
            f"Cannot find text module on {type(hf_model).__name__}")

    logger.info(
        "Fast contextual: sampling %d sentences, %d target tokens",
        n_sentences, len(token_ids),
    )

    sums = defaultdict(lambda: torch.zeros(d_model, dtype=torch.float32))
    counts = defaultdict(int)
    target_tensor = torch.tensor(list(target_set), device=device, dtype=torch.long)

    ds = load_dataset(
        "wikimedia/wikipedia", wikipedia_lang, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    n_processed = 0
    pending_sent_batch = []

    for row in ds:
        text = row["text"]
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ")]
        sentences = [
            s for s in sentences
            if min_sentence_len <= len(s) <= max_sentence_len
        ]

        for sent in sentences:
            if n_processed >= n_sentences:
                break
            pending_sent_batch.append(sent)

            if len(pending_sent_batch) >= batch_size:
                _process_sentence_batch(
                    tok, text_model, device, d_model,
                    pending_sent_batch, max_sentence_len,
                    target_set, target_tensor, sums, counts,
                )
                n_processed += len(pending_sent_batch)
                pending_sent_batch = []

                if n_processed % 5000 == 0:
                    n_hit = sum(1 for c in counts.values() if c > 0)
                    logger.info(
                        "  %d/%d sentences, %d tokens with >=1 context",
                        n_processed, n_sentences, n_hit)

        if n_processed >= n_sentences:
            break

    # Drain remaining
    if pending_sent_batch:
        _process_sentence_batch(
            tok, text_model, device, d_model,
            pending_sent_batch, max_sentence_len,
            target_set, target_tensor, sums, counts,
        )
        n_processed += len(pending_sent_batch)

    # Build output tensor
    X = torch.zeros(len(token_ids), d_model, dtype=torch.float32)
    for idx, tid in enumerate(token_ids):
        if counts[tid] > 0:
            X[idx] = sums[tid] / counts[tid]

    n_found = sum(1 for c in counts.values() if c > 0)
    avg_contexts = sum(counts.values()) / max(n_found, 1)
    logger.info(
        "Fast contextual done: %d/%d tokens found, "
        "%.1f avg contexts per token, %d sentences",
        n_found, len(token_ids), avg_contexts, n_processed,
    )

    return X


def _process_sentence_batch(
    tok, text_model, device, d_model,
    sent_batch, max_sentence_len,
    target_set, target_tensor, sums, counts,
):
    """Process a batch of sentences through the model and accumulate
    per-token hidden states.  (Helper for compute_contextual_fast.)"""
    if not sent_batch:
        return

    encoded = tok(sent_batch, padding=True, truncation=True,
                  max_length=max_sentence_len, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        outputs = text_model(input_ids)
        hidden = outputs.last_hidden_state.float()

    # GPU-side matching
    mask = torch.isin(input_ids, target_tensor)
    if mask.any():
        flat_mask = mask.reshape(-1)
        flat_ids = input_ids.reshape(-1)[flat_mask].cpu()
        flat_hidden = hidden.reshape(-1, d_model)[flat_mask].cpu()

        for tid, vec in zip(flat_ids.tolist(), flat_hidden):
            sums[tid] += vec
            counts[tid] += 1


# ---------------------------------------------------------------------------
# Eigendecomposition (shared by Option A and Option B)
# ---------------------------------------------------------------------------


def compute_similarity_eigenvectors(
    X: torch.Tensor,
    k: int = 20,
    center: bool = True,
) -> dict:
    """Compute top-k eigenvectors of the pairwise cosine similarity matrix.

    Uses the dual (Gram) approach: the d×d Gram matrix G = X̂ᵀX̂ is
    eigendecomposed, then left eigenvectors are projected back.

    Args:
        X: Token representations of shape (V, d) in float32.
        k: Number of top eigenvectors/eigenvalues to return.
        center: If True, center X (subtract mean) before L2-normalizing.
            This gives PCA of token variation. If False, use raw
            L2-normalized embeddings (absolute cosine similarity).

    Returns:
        Dict with keys:
            eigenvalues: (k,) float32 ndarray — top-k eigenvalues of S, descending.
            eigenvectors: (V, k) float32 ndarray — corresponding eigenvectors.
            explained_variance_ratio: (k,) fraction of total variance.
            singular_values: (d,) all singular values of X̂, descending.
            centered: bool.
            V: int.
            d: int.
    """
    V, d = X.shape

    if center:
        X = X - X.mean(dim=0, keepdim=True)

    # L2-normalize rows
    row_norms = X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    X_hat = X / row_norms  # (V, d)

    # Form Gram matrix G = X̂ᵀ X̂  (d × d)
    logger.info("Forming %d×%d Gram matrix...", d, d)
    G = X_hat.T @ X_hat  # (d, d)

    # Eigendecompose G (symmetric, PSD)
    logger.info("Running eigh on %d×%d Gram matrix...", d, d)
    eigvals_g, V_mat = torch.linalg.eigh(G)  # ascending eigenvalues

    # Convert to numpy for storage
    eigvals_g_np = eigvals_g.cpu().numpy()
    V_mat_np = V_mat.cpu().numpy()

    # Singular values of X̂: σ = √λ_g  (λ_g are eigenvalues of G = X̂ᵀX̂)
    s_vals = np.sqrt(np.maximum(eigvals_g_np, 0.0))

    # Eigenvalues of similarity matrix S = X̂X̂ᵀ are also λ_g (= σ²)
    eigenvalues = np.maximum(eigvals_g_np, 0.0)

    # Top-k: largest eigenvalues are at the end (ascending order)
    top_eigenvalues = eigenvalues[-k:][::-1]  # descending
    top_s_vals = s_vals[-k:][::-1]  # descending

    # Compute top-k left eigenvectors: U = X̂ V diag(1/σ)
    V_top = V_mat_np[:, -k:]  # (d, k), ascending
    V_top = V_top[:, ::-1]  # (d, k), descending

    X_hat_np = X_hat.cpu().numpy()
    U_top = X_hat_np @ V_top  # (V, k)
    # Normalize each column by 1/σ
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_s = np.where(top_s_vals > 1e-12, 1.0 / top_s_vals, 0.0)
    U_top = U_top * inv_s[np.newaxis, :]  # (V, k)

    # Explained variance ratio
    total_var = eigenvalues.sum()
    explained_var = top_eigenvalues / total_var if total_var > 0 else np.zeros(k)

    logger.info(
        "Top-%d eigenvalues: min=%.4f max=%.4f | "
        "explained variance: %.2f%% (top-1), %.2f%% (top-%d)",
        k,
        top_eigenvalues[-1],
        top_eigenvalues[0],
        100 * explained_var[0],
        100 * explained_var.sum(),
        k,
    )

    return {
        "eigenvalues": top_eigenvalues.astype(np.float32),
        "eigenvectors": U_top.astype(np.float32),
        "explained_variance_ratio": explained_var.astype(np.float32),
        "singular_values": s_vals[::-1].astype(np.float32),  # descending
        "centered": center,
        "V": V,
        "d": d,
    }


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------


def thought_manifold_pipeline(
    lens_model,
    token_ids: Optional[list[int]] = None,
    k: int = 20,
    batch_size: int = 512,
    center: bool = True,
    max_token_id: Optional[int] = None,
) -> dict:
    """Option A: singleton forward passes → Gram+eigh.

    Args:
        lens_model: HFLensModel wrapping the model.
        token_ids: Token IDs to compute (None = all vocab).
        k: Number of top eigenvectors.
        batch_size: Forward-pass batch size.
        center: Center embeddings before normalization.
        max_token_id: Max token ID when token_ids is None.

    Returns:
        Dict from compute_similarity_eigenvectors, plus:
            "representations": (V, d) float32 ndarray on CPU.
    """
    X = compute_token_representations(
        lens_model,
        token_ids=token_ids,
        batch_size=batch_size,
        max_token_id=max_token_id,
    )

    result = compute_similarity_eigenvectors(X, k=k, center=center)
    result["representations"] = X.cpu().numpy()
    return result


def thought_manifold_contextual_pipeline_fast(
    hf_model,
    token_ids: list[int],
    k: int = 50,
    n_contexts: int = 100,
    center: bool = True,
    sent_batch_size: int = 16,
) -> dict:
    """Option B: contextualized averaging → Gram+eigh.

    Args:
        hf_model: HuggingFace ``AutoModelForCausalLM`` already on GPU.
        token_ids: Target token IDs (from Wikipedia EN scan).
        k: Number of top eigenvectors.
        n_contexts: Target contexts per token.
        center: Center representations before normalization.
        sent_batch_size: Sentences per forward pass.

    Returns:
        Dict from compute_similarity_eigenvectors, plus:
            "representations": (V, d) float32 ndarray on CPU.
    """
    X = compute_contextual_fast(
        hf_model,
        token_ids=token_ids,
        n_sentences=n_contexts,
        batch_size=sent_batch_size,
    )

    result = compute_similarity_eigenvectors(X, k=k, center=center)
    result["representations"] = X.cpu().numpy()
    return result


def run_thought_manifold_for_checkpoint(
    model_id: str,
    step: int | str | None = None,
    revision: str | None = None,
    k: int = 20,
    batch_size: int = 512,
    center: bool = True,
    dtype: torch.dtype | str = "bfloat16",
    cache_dir: str | None = None,
    layout_str: str | None = None,
    contextual: bool = False,
    n_contexts: int = 100,
) -> dict:
    """Load a model, compute its thought manifold, and unload.

    When ``contextual=True`` (Option B), scans Wikipedia EN for token
    contexts instead of running singleton forward passes.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3.6-27B").
        step: Training step (revision) for Pythia-style checkpoints.
        revision: Direct revision/branch string for HF loading.
        k: Number of top eigenvectors.
        batch_size: Batch size (singleton passes, Option A only).
        center: Center representations before normalization.
        dtype: Torch dtype for model loading.
        cache_dir: HF cache directory.
        layout_str: Dotted path to text module.
        contextual: If True, use Option B (contextualized averaging).
        n_contexts: Target contexts per token (Option B only).

    Returns:
        Dict with manifold results + 'model_id'/'step'/'revision'.
    """
    upstream = get_upstream()

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    load_kwargs = dict(dtype=dtype)
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    if step is not None and revision is None:
        revision = f"step{step}"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading model %s (revision=%s, dtype=%s)...",
                model_id, revision, dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    try:
        if contextual:
            # Option B: determine token IDs from Wikipedia EN scan
            logger.info("Scanning Wikipedia EN for token frequency...")
            from collections import Counter
            from datasets import load_dataset
            import time

            t0 = time.time()
            token_counter = Counter()
            ds = load_dataset(
                "wikimedia/wikipedia", "20231101.en",
                split="train", streaming=True)
            n_chars = 0
            target_chars = 200_000_000
            for row in ds:
                ids = tokenizer.encode(row["text"],
                                       add_special_tokens=False)
                token_counter.update(ids)
                n_chars += len(row["text"])
                if n_chars >= target_chars:
                    break

            # Filter to tokens with >= n_contexts occurrences
            eligible = [tid for tid, cnt in token_counter.items()
                        if cnt >= n_contexts]
            eligible.sort()
            logger.info(
                "Wikipedia scan: %d unique tokens in %.1fM chars, "
                "%d eligible (>= %d contexts) in %.1fs",
                len(token_counter), n_chars / 1e6,
                len(eligible), n_contexts, time.time() - t0,
            )

            result = thought_manifold_contextual_pipeline(
                hf_model,
                token_ids=eligible,
                k=k,
                n_contexts=n_contexts,
                center=center,
            )
        else:
            # Option A: singleton passes
            layout = None
            if layout_str is not None:
                layout = upstream.Layout(path=layout_str)

            lens_model = upstream.from_hf(hf_model, tokenizer, layout=layout)
            logger.info("Detected layout: path=%s", lens_model.layout.path)
            logger.info("Wrapped as HFLensModel: n_layers=%d, d_model=%d",
                        lens_model.n_layers, lens_model.d_model)

            result = thought_manifold_pipeline(
                lens_model,
                k=k,
                batch_size=batch_size,
                center=center,
            )
            del lens_model

        result["model_id"] = model_id
        result["step"] = step
        result["revision"] = revision
        return result
    finally:
        del hf_model
        del tokenizer
        torch.cuda.empty_cache()
