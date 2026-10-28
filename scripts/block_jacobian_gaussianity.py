#!/usr/bin/env python3
"""Is the one-block linearization dh_{l+1}/dh_l a matrix of Gaussian entries?

Uses the upstream per-prompt estimator with target_layer = l + 1 (one block),
then tests J and J - I with jlens.gaussianity. Results go to a JSON file and an
npz with a small sample of standardized entries (for QQ plots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jlens._bridge import get_upstream
from jlens.checkpoint_manager import load_checkpoint
from jlens.gaussianity import remove_identity, summarize
from jlens.runner import _wrap_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="EleutherAI/pythia-12b-deduped")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1000, 143000])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 34])
    parser.add_argument("--prompts-file", type=Path, default=Path("prompts/overlap_prompts_seed123.txt"))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("figures/pythia-12b/block_jacobian_gaussianity.json"))
    args = parser.parse_args()

    prompt = args.prompts_file.read_text().splitlines()[args.prompt_index]
    upstream = get_upstream()
    results, samples = [], {}
    rng = np.random.default_rng(0)
    for step in args.steps:
        hf_model, tokenizer = load_checkpoint(args.model_id, step, dtype=torch.float16)
        lens_model = _wrap_model(hf_model, tokenizer)
        for layer in args.layers:
            jac, seq_len, n_valid = upstream.fitting.jacobian_for_prompt(
                lens_model, prompt, source_layers=[layer], target_layer=layer + 1, dim_batch=args.dim_batch
            )
            matrix = jac[layer].numpy().astype(np.float64)
            for variant, m in (("J", matrix), ("J_minus_I", remove_identity(matrix))):
                summary = summarize(m)
                summary.update(step=step, layer=layer, variant=variant, seq_len=seq_len, n_valid=n_valid)
                results.append(summary)
                z = (m.ravel() - m.mean()) / m.std()
                samples[f"{variant}_step{step}_layer{layer}"] = rng.choice(z, 50_000, replace=False).astype(np.float32)
                print(f"step {step} L{layer} {variant}: skew={summary['skew']:.2f} exkurt={summary['excess_kurtosis']:.2f} "
                      f"row_cv={summary['row_std_cv']:.3f} mp_ks={summary['mp_ks']:.3f} spikes={summary['n_above_edge']}", flush=True)
        del lens_model, hf_model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    np.savez_compressed(args.output.with_suffix(".npz"), **samples)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
