"""Compare upstream (Anthropic) vs blockwise (ours) Jacobian estimators.

Runs the upstream estimator on step143000 with 10 prompts, then compares
spectral metrics against the previously-computed blockwise results (1000 prompts).
"""
import os, time, json, h5py
import numpy as np
import torch

os.environ["TRITON_BUILD_WITHOUT_CC"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from jlens._bridge import get_upstream
from jlens.checkpoint_manager import load_checkpoint
from jlens.prompts import load_prompts_text
from jlens.accumulator import JacobianAccumulator
from jlens.spectra import analyze_checkpoint


def compute_metrics_from_eigvals(eigvals: np.ndarray) -> dict:
    """Compute coherence and effective rank from eigenvalues of Jbar^T Jbar."""
    total = eigvals.sum()
    if total <= 0:
        return {"coherence": 0.0, "effective_rank": 0.0, "fro_norm": 0.0, "nuc_norm": 0.0}
    p = eigvals / total
    p_pos = p[p > 0]
    eff_rk = float(np.exp(-np.sum(p_pos * np.log(p_pos))))
    fro = np.sqrt(np.sum(np.maximum(eigvals, 0)))
    nuc = np.sum(np.sqrt(np.maximum(eigvals, 0)))
    coh = fro / nuc if nuc > 0 else 0.0
    return {"coherence": coh, "effective_rank": eff_rk, "fro_norm": fro, "nuc_norm": nuc}


def main():
    # ---- Load old blockwise results (1000 prompts, all 12 layers) ----
    print("=== Loading old blockwise spectra ===", flush=True)
    with h5py.File("results/step143000/spectra.h5", "r") as f:
        old_eigvals = {}
        for layer_name in f.keys():
            li = int(layer_name.split("_")[1])
            old_eigvals[li] = f[layer_name]["eigenvalues"][:]

    print("Old blockwise (1000 prompts):")
    for li in sorted(old_eigvals.keys()):
        m = compute_metrics_from_eigvals(old_eigvals[li])
        print(f"  L{li:2d}: coh={m['coherence']:.4f}  eff_rk={m['effective_rank']:.0f}  "
              f"fro={m['fro_norm']:.1f}  nuc={m['nuc_norm']:.1f}")

    # ---- Run upstream estimator (10 prompts, layers 0,5,10) ----
    print()
    print("=== Running upstream estimator (10 prompts) ===", flush=True)

    t0 = time.time()
    hf_model, tokenizer = load_checkpoint(
        "EleutherAI/pythia-160m-deduped", 143000, dtype=torch.float16
    )
    up = get_upstream()
    layout = up.Layout(
        path="gpt_neox", layers="layers", norm="final_layer_norm",
        embed="embed_in", lm_head="lm_head",
    )
    lens_model = up.from_hf(hf_model, tokenizer, layout=layout)
    prompts = load_prompts_text(tokenizer, n_prompts=10, max_seq_len=128, seed=42)
    print(f"Setup: {time.time()-t0:.1f}s, {len(prompts)} prompts", flush=True)

    layers = [0, 5, 10]
    accum = JacobianAccumulator(lens_model.n_layers, lens_model.d_model, dtype=torch.float32)

    total_jac_time = 0.0
    for pi, text in enumerate(prompts):
        t0 = time.time()
        jacobians, seq_len, n_valid = up.fitting.jacobian_for_prompt(
            lens_model, text, source_layers=layers,
            dim_batch=128, max_seq_len=128, skip_first=16,
        )
        dt = time.time() - t0
        total_jac_time += dt
        accum.update(jacobians)
        print(f"  Prompt {pi+1:2d}: {dt:.1f}s  tok={n_valid}", flush=True)

    print(f"Total jacobian time: {total_jac_time:.1f}s "
          f"({total_jac_time/len(prompts):.1f}s/prompt)", flush=True)

    stats = accum.get_all_stats()
    spectral = analyze_checkpoint(stats)

    # ---- Comparison ----
    print()
    print("=" * 70)
    print("COMPARISON: Old blockwise (1000p) vs New upstream (10p)")
    print("=" * 70)
    print(f"{'Layer':<6} {'Metric':<16} {'Old (blockwise)':<18} {'New (upstream)':<18} {'Delta':<10}")
    print("-" * 70)

    for li in layers:
        old_m = compute_metrics_from_eigvals(old_eigvals[li])
        new_coh = spectral["coherence"][li]
        new_er = spectral["effective_rank"][li]
        tr_jtj = stats["tr_jtj_mean"][li]
        new_fro = np.sqrt(tr_jtj) if tr_jtj is not None and tr_jtj > 0 else float("nan")

        comparisons = [
            ("coherence", old_m["coherence"], new_coh),
            ("effective_rank", old_m["effective_rank"], new_er),
            ("fro_norm", old_m["fro_norm"], new_fro),
        ]
        for name, old_v, new_v in comparisons:
            if new_v is None or np.isnan(new_v):
                print(f"L{li:<5} {name:<16} {old_v:<18.4f} {'N/A':<18} ---")
                continue
            delta = new_v - old_v
            print(f"L{li:<5} {name:<16} {old_v:<18.4f} {new_v:<18.4f} {delta:+.4f}")

    # ---- Also compare Jbar directly (Frobenius distance) ----
    print()
    print("--- Jbar Frobenius distance (old vs new) ---")
    with h5py.File("results/step143000/stats.h5", "r") as f:
        for li in layers:
            if str(li) in f:
                old_jbar = torch.tensor(f[str(li)][:])  # shape (d, d)
                new_jbar = stats["jbar"][li]  # could be None
                if new_jbar is None:
                    print(f"  L{li}: no new Jbar")
                    continue
                new_jbar_t = torch.tensor(new_jbar)
                fro_dist = (old_jbar - new_jbar_t).norm("fro").item()
                rel_dist = fro_dist / old_jbar.norm("fro").item()
                cos_sim = (old_jbar.flatten() @ new_jbar_t.flatten()).item() / (
                    old_jbar.norm().item() * new_jbar_t.norm().item()
                )
                print(f"  L{li}: fro_dist={fro_dist:.4f}  rel_dist={rel_dist:.4f}  "
                      f"cos_sim={cos_sim:.6f}")

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    main()
