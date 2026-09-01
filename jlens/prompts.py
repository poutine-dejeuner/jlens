"""Wikitext prompt loading."""

import logging
from typing import Iterator

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
        # Remove batch dimension
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}

        # Skip very short sequences
        if tokens["input_ids"].shape[0] < 4:
            continue

        prompts.append(tokens)

        if len(prompts) >= n_prompts:
            break

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

        # Quick length check via tokenizer to skip very short texts
        # Upstream jlens requires seq_len > skip_first + 1 (default skip_first=16 -> 17 min)
        n_tokens = len(tokenizer.encode(text, truncation=True, max_length=max_seq_len + 1))
        if n_tokens < 18:
            continue

        prompts.append(text)

        if len(prompts) >= n_prompts:
            break

    logger.info(f"Loaded {len(prompts)} text prompts")
    return prompts
