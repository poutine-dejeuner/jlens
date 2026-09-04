"""Wikitext prompt loading."""

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
