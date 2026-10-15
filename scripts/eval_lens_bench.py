#!/usr/bin/env python3
"""Evaluate Pythia-12B-deduped checkpoints on all 6 J-lens benchmarks.

Method (matching Denis's jx/evals.py):
  - For each benchmark item, apply the stored J-lens to the readout position
  - Compute the min-over-layers rank of each intermediate concept token
  - Score: pass@k = fraction of intermediates with rank ≤ k

All 6 benchmarks use this method — no next-token generation needed.
J-lens matrices are loaded from our HDF5 storage.

Output:
  {results_dir}/lens_bench/step{step}.json     — per-checkpoint results
  {results_dir}/lens_bench/all_results.json     — aggregate
"""
import json, argparse
from pathlib import Path
import torch
import numpy as np
import h5py
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm

BENCHMARK_NAMES = ["multihop", "multilingual", "order-ops", "association", "typo", "poetry"]
UPSTREAM = Path("/mnt/home/jacob/vl/repos/anthropic/jacobian-lens")

# Readout rule per benchmark (matching Denis's SETS)
READOUT_RULES = {
    "multihop": "last",
    "multilingual": "last",
    "order-ops": "last",
    "association": "last",
    "typo": "last",
    "poetry": "last-newline",
}
PASS_AT = (1, 5, 10, 50)

_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    24: "twenty-four", 30: "thirty", 40: "forty", 50: "fifty",
}
_OP_FORMS = {
    "addition": ["addition", "add", "plus", "+"],
    "subtraction": ["subtraction", "subtract", "minus", "-"],
    "multiplication": ["multiplication", "multiply", "times", "*"],
    "division": ["division", "divide", "/"],
    "mod": ["mod", "modulo", "%", "remainder"],
    "squared": ["squared", "square", "^", "**"],
}


def surface_forms(word: str, set_name: str) -> list[str]:
    """All surface strings tried for one intermediate (matching Denis's)."""
    base = [word]
    if set_name == "order-ops":
        if word.lstrip("-").isdigit():
            n = int(word)
            base = [word] + ([_NUMBER_WORDS[n]] if n in _NUMBER_WORDS else [])
        else:
            base = _OP_FORMS.get(word, [word])
    forms = []
    for w in base:
        for v in (w, w.lower(), w.capitalize()):
            for prefixed in (" " + v, v):
                if prefixed not in forms:
                    forms.append(prefixed)
    return forms


def candidate_token_ids(tokenizer, word: str, set_name: str) -> tuple[list[int], bool]:
    """First-token ids of every surface form, plus coverage check."""
    ids = []
    for form in surface_forms(word, set_name):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if encoded and encoded[0] not in ids:
            ids.append(encoded[0])
    primary = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids, len(primary) == 1


def readout_position(tokenizer, prompt: str, rule: str, max_seq_len: int = 512) -> int:
    """Token index to read at (matching Denis's implementation)."""
    encoding = tokenizer(
        prompt, truncation=True, max_length=max_seq_len, return_offsets_mapping=True
    )
    n = len(encoding["input_ids"])
    if rule == "last":
        return n - 1
    if rule == "last-newline":
        char = prompt.rfind("\n")
        if char < 0:
            return n - 1
        for idx in range(n - 1, -1, -1):
            start, end = encoding["offset_mapping"][idx]
            if start <= char < end and end > start:
                return idx
        return n - 1
    raise ValueError(rule)


def load_benchmarks() -> dict:
    """Return {name: [items]} for all 6 benchmarks."""
    benchmarks = {}
    for name in BENCHMARK_NAMES:
        path = UPSTREAM / "data" / "evaluations" / f"lens-eval-{name}.json"
        with open(path) as f:
            data = json.load(f)
        benchmarks[name] = data["items"]
    return benchmarks


def load_jbar(results_dir: Path, step: int) -> dict[int, torch.Tensor]:
    """Load J-bar matrices for all layers from HDF5. Returns {layer_idx: tensor}."""
    step_dir = results_dir / f"step{step:04d}"
    jbar = {}
    with h5py.File(step_dir / "stats.h5", "r") as f:
        for key in f.keys():
            if not key.startswith("layer_"):
                continue
            li = int(key.split("_")[1])
            lg = f[key]
            if "jbar" in lg:
                jbar[li] = torch.from_numpy(lg["jbar"][:])
    return jbar


def evaluate_checkpoint(
    model, tokenizer, jbars, benchmarks: dict, device,
    max_seq_len: int = 512, batch_size: int = 32,
) -> dict:
    """Evaluate all 6 benchmarks on one checkpoint.

    Batched: runs all prompts through the model in bulk, then applies
    J-lens per layer per readout position.
    """
    # Flatten all items into one list, annotated with set_name
    all_items = []
    for set_name in BENCHMARK_NAMES:
        for item in benchmarks[set_name]:
            all_items.append((set_name, item))

    # Phase 1: tokenize all prompts and collect readout positions
    all_prompts = []
    all_positions = []
    all_set_names = []

    for set_name, item in all_items:
        rule = READOUT_RULES[set_name]
        pos = readout_position(tokenizer, item["prompt"], rule, max_seq_len=max_seq_len)
        all_prompts.append(item["prompt"])
        all_positions.append(pos)
        all_set_names.append(set_name)

    # Phase 2: batch tokens — pad to max length in batch
    encoded = tokenizer(
        all_prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_seq_len,
    ).to(device)

    # Phase 3: single forward pass with hidden states
    print(f"  Forward pass: {len(all_prompts)} prompts, "
          f"shape={encoded['input_ids'].shape}...", flush=True)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_model)

    # Phase 4: apply J-lens per layer per item per readout position
    lm_head = model.get_output_embeddings()
    sorted_layers = sorted(jbars.keys())
    n_layers = len(sorted_layers)
    n_items = len(all_prompts)
    vocab_size = lm_head.weight.shape[0]

    # Move jbars to device once
    jbars_dev = {li: jbar.to(device=device, dtype=hidden_states[1].dtype)
                 for li, jbar in jbars.items()}

    # For each layer, compute lens logits for all items at their readout positions
    # Store per-item, per-layer logits: list of dicts
    item_logits = [{} for _ in range(n_items)]

    for li in tqdm(sorted_layers, desc="  J-lens layers"):
        jbar = jbars_dev[li]
        # hidden_states[0] = embedding, hidden_states[1] = layer 0 output, etc.
        h = hidden_states[li + 1]  # (batch, seq, d_model)

        for idx in range(n_items):
            pos = all_positions[idx]
            hs = h[idx, pos]  # (d_model,)
            transported = jbar @ hs  # (d_model,)
            l = lm_head(transported.unsqueeze(0))[0]  # (vocab,)
            item_logits[idx][li] = l

    # Phase 5: score each item — per-intermediate ranks, all_items x max_interms
    max_interms = max(len(item["intermediates"]) for _, item in all_items)
    all_ranks = np.full((n_items, max_interms), np.nan)
    coverage_by_set = {name: {"covered": 0, "total": 0} for name in BENCHMARK_NAMES}

    for idx, (set_name, item) in enumerate(tqdm(all_items, desc="  Scoring")):
        for j, word in enumerate(item["intermediates"][:max_interms]):
            ids, is_single = candidate_token_ids(tokenizer, word, set_name)
            coverage_by_set[set_name]["covered"] += is_single
            coverage_by_set[set_name]["total"] += 1
            best_rank = None
            for li, logits in item_logits[idx].items():
                for tid in ids:
                    rank = int((logits > logits[tid]).sum().item()) + 1
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
            if best_rank is not None:
                all_ranks[idx, j] = best_rank

    # Final scoring: pass@k = mean over items of fraction of intermediates ≤ k
    # (matching Denis's jx/evals.py: nanmean(ranks<=k, axis=1) then nanmean)
    final = {}
    set_indices = {}  # {set_name: list of row indices in all_ranks}
    for idx, (set_name, _) in enumerate(all_items):
        set_indices.setdefault(set_name, []).append(idx)

    for set_name in BENCHMARK_NAMES:
        rows = set_indices.get(set_name, [])
        sub = all_ranks[rows]  # (n_items_in_set, max_interms)
        finite_all = sub[np.isfinite(sub)]
        final[set_name] = {
            "n_items": len(benchmarks[set_name]),
            "coverage": round(
                coverage_by_set[set_name]["covered"]
                / max(coverage_by_set[set_name]["total"], 1), 3
            ),
            "pass_at": {
                k: round(float(np.nanmean(
                    np.nanmean(np.where(np.isfinite(sub), sub <= k, np.nan), axis=1)
                )), 4)
                for k in PASS_AT
            },
            "median_rank": float(np.median(finite_all)) if len(finite_all) else None,
        }

    return final


def discover_checkpoints(results_dir: Path) -> list[int]:
    """Find all completed checkpoints with stats.h5.

    Returns a list of integer step numbers.  Also returns the widest zero-padding
    found so we can reconstruct directory names later.
    """
    steps = []
    for d in results_dir.iterdir():
        if d.name.startswith("step") and (d / "stats.h5").exists():
            try:
                steps.append(int(d.name[4:]))
            except ValueError:
                pass
    steps.sort()
    return steps


def evaluate_one_step(
    step: int, model_name: str, model_cache: str,
    results_dir: Path, benchmarks: dict[str, list], dtype,
    max_seq_len: int = 512,
) -> dict:
    """Load model checkpoint, run J-lens eval, return results dict."""
    device_map = "auto" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"Loading {model_name} step{step}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=f"step{step}", cache_dir=model_cache
    )
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name, revision=f"step{step}",
        cache_dir=model_cache, torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.output_hidden_states = True
    model.eval()
    print(f"  Model on device: {model.device}", flush=True)

    jbars = load_jbar(results_dir, step)
    print(f"  {len(jbars)} layers loaded", flush=True)

    results = evaluate_checkpoint(
        model, tokenizer, jbars, benchmarks, model.device, max_seq_len=max_seq_len
    )

    del model, tokenizer, jbars
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--model-name", default="EleutherAI/pythia-12b-deduped")
    parser.add_argument("--model-cache", default="/mnt/home/jacob/.cache/huggingface/hub")
    parser.add_argument("--step", type=int, default=None, help="Single checkpoint step to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints with stats.h5")
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "lens_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16
    benchmarks = load_benchmarks()
    total_items = sum(len(items) for items in benchmarks.values())
    print(f"Loaded {total_items} items across {len(benchmarks)} benchmarks", flush=True)

    if args.all:
        steps = discover_checkpoints(results_dir)
        print(f"Found {len(steps)} checkpoints with stats.h5", flush=True)

        for i, step in enumerate(steps):
            out_path = out_dir / f"step{step:04d}.json"
            if out_path.exists():
                print(f"[{i+1}/{len(steps)}] step{step} — already done, skipping", flush=True)
                continue
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(steps)}] step{step}", flush=True)
            print(f"{'='*60}")

            try:
                results = evaluate_one_step(
                    step, args.model_name, args.model_cache,
                    results_dir, benchmarks, dtype, max_seq_len=args.max_seq_len
                )
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                continue

            for name, r in results.items():
                print(f"  {name:15s} coverage={r['coverage']:.2f} "
                      f"pass@1={r['pass_at'][1]:.3f} pass@10={r['pass_at'][10]:.3f} "
                      f"median_rank={r['median_rank']}")

            output = {"step": step, "benchmarks": results}
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Saved to {out_path}")
    else:
        step = args.step or max(discover_checkpoints(results_dir))
        results = evaluate_one_step(
            step, args.model_name, args.model_cache,
            results_dir, benchmarks, dtype, max_seq_len=args.max_seq_len
        )

        for name, r in results.items():
            print(f"  {name:15s} coverage={r['coverage']:.2f} "
                  f"pass@1={r['pass_at'][1]:.3f} pass@10={r['pass_at'][10]:.3f} "
                  f"median_rank={r['median_rank']}")

        output = {"step": step, "benchmarks": results}
        with open(out_dir / f"step{step:04d}.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_dir}/step{step:04d}.json")


if __name__ == "__main__":
    main()
