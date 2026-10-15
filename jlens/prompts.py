"""Prompt loading: wikitext and random-token generation."""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_prompts(
    tokenizer: PreTrainedTokenizer,
    n_prompts: int = 1000,
    max_seq_len: int = 256,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    dataset_split: str = "test",
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Load tokenized prompts from wikitext.

    Returns:
        List of dicts with 'input_ids' and 'attention_mask' tensors.
    """
    logger.info(f"Loading {n_prompts} prompts from {dataset_name} ({dataset_config})")

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

    prompts = []
    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}

        if tokens["input_ids"].shape[0] < 4:
            continue

        prompts.append(tokens)

    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)
    prompts = prompts[:n_prompts]

    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def load_prompts_text(
    tokenizer: PreTrainedTokenizer,
    n_prompts: int = 1000,
    max_seq_len: int = 256,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    dataset_split: str = "test",
    seed: int = 42,
) -> list[str]:
    """Load raw text prompts from wikitext (for upstream jlens estimator).

    Upstream jlens.jacobian_for_prompt handles tokenization internally,
    so we just return the raw text.

    Returns:
        List of raw text strings.
    """
    logger.info(f"Loading {n_prompts} text prompts from {dataset_name} ({dataset_config})")

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

    prompts = []
    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue

        n_tokens = len(tokenizer.encode(text, truncation=True, max_length=max_seq_len + 1))
        if n_tokens < 18:
            continue

        prompts.append(text)

    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)
    prompts = prompts[:n_prompts]

    logger.info(f"Loaded {len(prompts)} text prompts")
    return prompts


def generate_random_prompts(
    tokenizer: PreTrainedTokenizer,
    n_prompts: int = 1000,
    seq_len: int = 64,
    seed: int = 42,
) -> list[str]:
    """Generate prompts from uniformly random token sequences.

    Each prompt is seq_len random token IDs drawn uniformly from
    [0, vocab_size), decoded to text via the tokenizer.  The upstream
    jlens estimator re-tokenizes internally, so the decode/encode
    round-trip may not be identity for all tokenizers, but the
    resulting sequences remain uniformly random.

    Args:
        tokenizer: HuggingFace tokenizer (for vocab_size and decoding).
        n_prompts: Number of prompts to generate.
        seq_len: Token sequence length per prompt.
        seed: Random seed for reproducibility.

    Returns:
        List of raw text strings.
    """
    vocab_size = tokenizer.vocab_size
    rng = np.random.default_rng(seed)

    # Exclude special tokens that may produce empty decode or BOS/EOS noise.
    # We bias toward 'normal' token IDs: skip 0 (often unk/pad) and any
    # special-token range at the high end.
    special_ids = set(tokenizer.all_special_ids or [])
    # Also exclude 0 (unk/pad in GPT-NeoX) and < bos>/<eos> if known
    exclude = {0} | special_ids
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        exclude.add(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        exclude.add(tokenizer.eos_token_id)

    valid_ids = [i for i in range(vocab_size) if i not in exclude]
    if not valid_ids:
        valid_ids = list(range(1, vocab_size))  # fallback: exclude only 0

    logger.info(
        f"Generating {n_prompts} random prompts: seq_len={seq_len}, "
        f"vocab_size={vocab_size}, valid_ids={len(valid_ids)}, seed={seed}"
    )

    prompts: list[str] = []
    for _ in range(n_prompts):
        ids = rng.choice(valid_ids, size=seq_len, replace=True).tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if not text.strip():
            # Rare: all tokens decoded to whitespace.  Force a single
            # printable token at position 0 as a fallback.
            ids[0] = valid_ids[0]
            text = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(text)

    logger.info(f"Generated {len(prompts)} random-text prompts")
    return prompts


def save_prompts_text(
    prompts: list[str],
    seed: int,
    out_dir: str | Path = "prompts",
) -> Path:
    """Save a list of text prompts with seed annotation for reproducibility.

    Writes:
        prompts/overlap_prompts_seed{seed}.txt  — one prompt per line
        prompts/overlap_prompts_manifest.json   — metadata (seed, count)

    Returns path to the text file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"overlap_prompts_seed{seed}.txt"
    with open(txt_path, "w") as f:
        for p in prompts:
            f.write(p.replace("\n", " ") + "\n")

    manifest_path = out_dir / "overlap_prompts_manifest.json"
    import json
    with open(manifest_path, "w") as f:
        json.dump({"seed": seed, "n_prompts": len(prompts)}, f, indent=2)

    logger.info(f"Saved {len(prompts)} overlap prompts to {txt_path}")
    return txt_path
