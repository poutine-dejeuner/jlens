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
    rng = torch.Generator().manual_seed(seed)

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
