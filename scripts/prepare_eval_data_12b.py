#!/usr/bin/env python3
"""Prepare 12B evaluation data for J-lens correlation analysis.

1. Downloads any missing EleutherAI eval JSONs for Pythia-12B steps.
2. Extracts benchmark scores into a matrix.
3. Cross-joins with J-lens metrics from /data/jlens/pythia-12b-deduped/step*/stats.h5 + spectra.h5.
4. Saves to eval_data/12b_cross_join.npz for plotting.

Usage:
    uv run python scripts/prepare_eval_data_12b.py
"""

import json
import os
import urllib.request

import h5py
import numpy as np
from scipy import stats

# EleutherAI eval JSON URL pattern for Pythia-12B
REPO_BASE = "https://raw.githubusercontent.com/EleutherAI/pythia/main"
EVAL_PATH = "evals/pythia-v1/pythia-12b-deduped/zero-shot"

# All 27 steps evaluated by EleutherAI for 12B
ELEUTHER_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000,
    13000, 23000, 33000, 43000, 53000, 63000, 73000, 83000,
    93000, 103000, 113000, 123000, 133000, 143000,
]

# All 48 J-lens 12B steps
JLENS_STEPS = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    2000, 5000, 9000,
    13000, 17000, 21000, 24000, 28000, 32000, 36000, 40000, 43000,
    47000, 51000, 55000, 59000, 62000, 66000, 70000, 74000, 78000,
    82000, 85000, 89000, 93000, 97000,
    101000, 104000, 108000, 112000, 116000, 120000, 123000,
    127000, 131000, 135000, 139000, 143000,
]

KEY_BENCHMARKS = [
    "arc_easy", "arc_challenge", "piqa", "winogrande",
    "sciq", "lambada_openai", "wsc", "logiqa",
]

# 12B results directory
RESULTS_DIR = "/data/jlens/pythia-12b-deduped"


def download_missing_evals(out_dir: str = "eval_data/12b_zero_shot") -> list[int]:
    """Download any missing EleutherAI eval JSONs."""
    os.makedirs(out_dir, exist_ok=True)
    downloaded = []

    for step in ELEUTHER_STEPS:
        fname = f"12b-deduped_step{step}.json"
        url = f"{REPO_BASE}/{EVAL_PATH}/{fname}"
        out_path = os.path.join(out_dir, f"step{step}.json")

        if os.path.exists(out_path):
            downloaded.append(step)
            continue

        print(f"  downloading step {step}...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.load(resp)
            with open(out_path, "w") as f:
                json.dump(data, f)
            downloaded.append(step)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")

    return downloaded


def extract_scores(eval_dir: str = "eval_data/12b_zero_shot") -> dict:
    """Parse all eval JSONs into a step × benchmark matrix.

    Supports both EleutherAI format (full results dict) and our simplified format
    (just {'results': {task: {'acc': ..., 'acc_norm': ...}}}).
    """
    scores = {bm: {} for bm in KEY_BENCHMARKS}
    steps = []

    for fname in sorted(os.listdir(eval_dir)):
        step = int(fname.replace("step", "").replace(".json", ""))
        steps.append(step)
        with open(os.path.join(eval_dir, fname)) as f:
            data = json.load(f)

        # Handle both EleutherAI full format and our simplified format
        results = data.get("results", {})
        for bm in KEY_BENCHMARKS:
            if bm in results:
                res = results[bm]
                # lm_eval puts acc directly, or nested
                if isinstance(res, dict):
                    if bm == "lambada_openai":
                        scores[bm][step] = res.get("acc", np.nan)
                    else:
                        scores[bm][step] = res.get("acc", res.get("acc_norm", np.nan))
                else:
                    # Might be a raw number
                    scores[bm][step] = float(res)

    return {"steps": sorted(steps), "scores": scores}


def safe_mean(arr: list, n: int = 36) -> float:
    """Mean of first n elements, ignoring NaN."""
    vals = [x for i, x in enumerate(arr) if i < n and not np.isnan(x)]
    return float(np.mean(vals)) if vals else np.nan


def extract_jlens_metrics(
    results_dir: str = RESULTS_DIR,
    eval_steps: list[int] | None = None,
) -> dict:
    """Cross-join J-lens metrics for matching evaluation steps."""
    metrics = {
        "coherence_L0": [], "coherence_L5": [], "coherence_mean": [],
        "a_coeff_L0": [], "a_coeff_L5": [], "a_coeff_mean": [],
        "eff_rank_L0": [], "eff_rank_L5": [], "eff_rank_mean": [],
        "tr_jtj_L0": [], "tr_jtj_L5": [], "tr_jtj_mean": [],
        "alpha_L0": [], "alpha_L5": [], "alpha_mean": [],
        "n_spikes_L0": [], "n_spikes_L5": [], "n_spikes_mean": [],
    }

    matched_steps = []

    for step in (eval_steps or JLENS_STEPS):
        if step == 0:
            continue
        d = os.path.join(results_dir, f"step{step:04d}")
        stats_path = os.path.join(d, "stats.h5")
        spectra_path = os.path.join(d, "spectra.h5")

        if not os.path.exists(stats_path) or not os.path.exists(spectra_path):
            print(f"  missing J-lens data for step {step}")
            continue

        matched_steps.append(step)

        with (
            h5py.File(stats_path, "r") as fs,
            h5py.File(spectra_path, "r") as fsp,
        ):
            coh_vals, a_vals, tr_vals, alpha_vals, er_vals, spike_vals = (
                [], [], [], [], [], []
            )
            for li in range(36):  # 12B has 36 layers
                key = f"layer_{li}"
                if key in fs:
                    attrs = dict(fs[key].attrs)
                    coh_vals.append(float(attrs.get("coherence", np.nan)))
                    a_vals.append(float(attrs.get("a_coeff", np.nan)))
                    tr_vals.append(float(attrs.get("tr_jtj_mean", np.nan)))
                else:
                    coh_vals.append(np.nan)
                    a_vals.append(np.nan)
                    tr_vals.append(np.nan)

                if key in fsp and "eigenvalues" in fsp[key]:
                    ev = fsp[key]["eigenvalues"][:]
                    if ev is not None and len(ev) > 0:
                        ev_pos = np.maximum(ev, 1e-15)
                        ev_norm = ev_pos / ev_pos.sum()
                        entropy = -np.sum(ev_norm * np.log(ev_norm))
                        er_vals.append(np.exp(entropy))
                        alpha = fsp[key].attrs.get("power_law_alpha", np.nan)
                        alpha_vals.append(float(alpha))
                        q = 1000.0 / 768
                        bulk_edge = (1 + np.sqrt(q)) ** 2
                        spike_vals.append(int(np.sum(ev > 3 * bulk_edge)))
                    else:
                        er_vals.append(np.nan)
                        alpha_vals.append(np.nan)
                        spike_vals.append(0)
                else:
                    er_vals.append(np.nan)
                    alpha_vals.append(np.nan)
                    spike_vals.append(0)

        metrics["coherence_L0"].append(coh_vals[0])
        metrics["coherence_L5"].append(coh_vals[5])
        metrics["coherence_mean"].append(safe_mean(coh_vals))
        metrics["a_coeff_L0"].append(a_vals[0])
        metrics["a_coeff_L5"].append(a_vals[5])
        metrics["a_coeff_mean"].append(safe_mean(a_vals))
        metrics["eff_rank_L0"].append(er_vals[0])
        metrics["eff_rank_L5"].append(er_vals[5])
        metrics["eff_rank_mean"].append(safe_mean(er_vals))
        metrics["tr_jtj_L0"].append(tr_vals[0])
        metrics["tr_jtj_L5"].append(tr_vals[5])
        metrics["tr_jtj_mean"].append(safe_mean(tr_vals))
        metrics["alpha_L0"].append(alpha_vals[0])
        metrics["alpha_L5"].append(alpha_vals[5])
        metrics["alpha_mean"].append(safe_mean(alpha_vals))
        metrics["n_spikes_L0"].append(spike_vals[0])
        metrics["n_spikes_L5"].append(spike_vals[5])
        metrics["n_spikes_mean"].append(safe_mean([float(s) for s in spike_vals]))

    return {"matched_steps": matched_steps, "metrics": metrics}


def main():
    print("Step 1: Downloading any missing EleutherAI eval JSONs for 12B...")
    download_missing_evals()
    eval_files = [f for f in os.listdir("eval_data/12b_zero_shot") if f.endswith(".json")]
    print(f"  {len(eval_files)} eval files available")

    print("\nStep 2: Extracting benchmark scores...")
    eval_data = extract_scores()
    eval_steps_avail = eval_data["steps"]
    scores = eval_data["scores"]
    print(f"  {len(eval_steps_avail)} checkpoints with eval data")

    # Determine which steps to cross-join: intersection of J-lens and eval data
    # Also include steps from our newly computed eval results
    # Find all eval steps that have matching J-lens data
    cross_steps = []
    for step in sorted(eval_steps_avail + JLENS_STEPS):
        if step == 0:
            continue
        step_dir = os.path.join(RESULTS_DIR, f"step{step:04d}")
        if os.path.exists(os.path.join(step_dir, "stats.h5")):
            if step not in cross_steps:
                cross_steps.append(step)
    cross_steps = sorted(cross_steps)

    print(f"\nStep 3: Cross-joining with J-lens metrics ({len(cross_steps)} checkpoints)...")
    jlens_data = extract_jlens_metrics(eval_steps=cross_steps)
    matched = jlens_data["matched_steps"]
    metrics = jlens_data["metrics"]
    print(f"  {len(matched)} matched checkpoints with complete J-lens data")

    # Merge in eval scores from both EleutherAI and our computed results
    print("\nStep 4: Saving 12b_cross_join.npz...")
    out = {
        "steps": matched,
        "benchmarks": KEY_BENCHMARKS,
    }
    for bm in KEY_BENCHMARKS:
        out[f"eval_{bm}"] = np.array(
            [scores[bm].get(s, np.nan) for s in matched]
        )
    for k, v in metrics.items():
        out[f"jlens_{k}"] = np.array(v)

    os.makedirs("eval_data", exist_ok=True)
    np.savez("eval_data/12b_cross_join.npz", **out)

    # Print summary of how many have eval data vs missing
    n_with_eval = sum(1 for s in matched if any(
        not np.isnan(scores[bm].get(s, np.nan)) for bm in KEY_BENCHMARKS
    ))
    print(f"  {n_with_eval}/{len(matched)} checkpoints have evaluation scores")

    print("\nTop correlations (|ρ| > 0.80):")
    for mkey in sorted(metrics.keys()):
        mvals = np.array(metrics[mkey])
        for bm in KEY_BENCHMARKS:
            bvals = np.array([scores[bm].get(s, np.nan) for s in matched])
            mask = ~np.isnan(mvals) & ~np.isnan(bvals)
            if mask.sum() >= 5:
                r, p = stats.spearmanr(mvals[mask], bvals[mask])
                if abs(r) > 0.80:
                    sig = "**" if p < 0.01 else "*"
                    print(f"  ρ = {r:+.4f}{sig}  |  {mkey:25s}  ×  {bm} (n={mask.sum()})")

    print("\nDone. Run `uv run python scripts/plot_benchmark_corr_12b.py` for figures.")


if __name__ == "__main__":
    main()
