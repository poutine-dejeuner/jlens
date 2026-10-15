#!/usr/bin/env python3
"""Extract token embedding point clouds from Pythia-12B for TDA dimension analysis.

Output:
  {out_dir}/singleton_embeddings.npy   — embedding layer weight matrix (vocab_size, d)
  {out_dir}/context_layer{N}.npy       — context embeddings at layer N (n_tokens, d)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_singleton_embeddings(model, out_path: Path):
    """Extract the embedding layer weight (token embeddings without context)."""
    wte = model.get_input_embeddings()
    embeddings = wte.weight.detach().float().cpu().numpy()
    np.save(out_path, embeddings)
    print(f"  Singleton embeddings: {embeddings.shape} → {out_path}")


def extract_context_embeddings(model, tokenizer, prompts: list[str], out_dir: Path):
    """Run forward pass and capture hidden states at each layer."""
    device = next(model.parameters()).device

    # Tokenize all prompts, pad to longest
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=64,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print(f"  Tokenized: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, d)
    hidden_states = outputs.hidden_states

    for layer_idx, hs in enumerate(hidden_states):
        # Flatten batch × seq_len, but only keep non-padding tokens
        mask = attention_mask.bool().cpu()
        hs_valid = hs[mask].cpu().float().numpy()
        out_path = out_dir / f"context_layer{layer_idx}.npy"
        np.save(out_path, hs_valid)
        print(f"  Layer {layer_idx}: {hs_valid.shape} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-12b-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--prompts", default="prompts/overlap_prompts_seed123.txt")
    parser.add_argument("--max-prompts", type=int, default=100)
    parser.add_argument("--out-dir", default="/data/jlens/pythia-12b-deduped/embeddings_step143000")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model} @ {args.revision}")
    print(f"Output: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True,
    )

    # 1. Singleton embeddings (no context)
    print("\n=== Singleton (embedding layer weight) ===")
    extract_singleton_embeddings(model, out_dir / "singleton_embeddings.npy")

    # 2. Context embeddings from prompts
    print("\n=== Context (hidden states from prompts) ===")
    if Path(args.prompts).exists():
        with open(args.prompts) as f:
            all_prompts = [line.strip() for line in f if line.strip()]
        prompts = all_prompts[:args.max_prompts]
        print(f"  {len(prompts)} prompts from {args.prompts}")
    else:
        prompts = [
            "The capital of France is",
            "Machine learning is a subfield of",
            "Once upon a time there was",
            "In quantum mechanics, the wave function",
            "The theory of relativity states that",
            "Deep neural networks can approximate",
            "The mitochondria is the powerhouse of",
            "Language models are trained to predict",
            "The Pythagorean theorem relates",
            "Gravity is a fundamental force that",
        ] * 10
        prompts = prompts[:args.max_prompts]
        print(f"  {len(prompts)} default prompts")

    extract_context_embeddings(model, tokenizer, prompts, out_dir)

    print(f"\nDone. Files in {out_dir}/")


if __name__ == "__main__":
    main()
