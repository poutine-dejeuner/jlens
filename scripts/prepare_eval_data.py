#!/usr/bin/env python3
"""Prepare evaluation data for J-lens correlation analysis.

Pipeline:
  1. Download Pythia-160M-deduped eval JSONs from EleutherAI/pythia.
  2. Extract benchmark scores into a matrix.
  3. Cross-join with J-lens metrics from results/step*/stats.h5 and spectra.h5.
  4. Save to eval_data/cross_join.npz for plotting.

Usage:
    python scripts/prepare_eval_data.py

Requires: existing J-lens results in results/step*/
"""

import json
import os
import re
import urllib.request

import h5py
import numpy as np
from scipy import stats

REPO_BASE = "https://raw.githubusercontent.com/EleutherAI/pythia/main"
EVAL_PATH = "evals/pythia-v1/pythia-160m-deduped/zero-shot"

STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000,
    13000, 23000, 33000, 43000, 53000, 63000, 73000, 83000,
    93000, 103000, 113000, 123000, 133000, 143000,
]

KEY_BENCHMARKS = [
    "arc_easy", "arc_challenge", "piqa", "winogrande",
    "sciq", "lambada_openai", "logiqa", "wsc",
]


def download_evals(out_dir: str = "eval_data/zero_shot") -> list[int]:
    """Download zero-shot eval JSONs, return list of successfully downloaded steps."""
    os.makedirs(out_dir, exist_ok=True)
    downloaded = []

    for step in STEPS:
        filename = f"160m-deduped_step{step}.json"
        url = f"{REPO_BASE}/{EVAL_PATH}/{filename}"
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


def extract_scores(eval_dir: str = "eval_data/zero_shot") -> dict:
    """Parse all eval JSONs into a step × benchmark matrix."""
    scores = {bm: {} for bm in KEY_BENCHMARKS}
    steps = []

    for fname in sorted(os.listdir(eval_dir)):
        step = int(fname.replace("step", "").replace(".json", ""))
        steps.append(step)
        with open(os.path.join(eval_dir, fname)) as f:
            data = json.load(f)
        for bm in KEY_BENCHMARKS:
            if bm in data["results"]:
                res = data["results"][bm]
                if bm == "lambada_openai":
                    scores[bm][step] = res.get("acc", np.nan)
                else:
                    scores[bm][step] = res.get("acc", res.get("acc_norm", np.nan))

    return {"steps": sorted(steps), "scores": scores}


def safe_mean(arr: list, n: int = 11) -> float:
    """Mean of first n elements, ignoring NaN."""
    vals = [x for i, x in enumerate(arr) if i < n and not np.isnan(x)]
    return float(np.mean(vals)) if vals else np.nan


def extract_jlens_metrics(
    results_dir: str = "results",
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

    for step in (eval_steps or STEPS):
        if step == 0:
            continue  # No J-lens data for step 0
        d = os.path.join(results_dir, f"step{step:04d}")
        if not os.path.exists(os.path.join(d, "stats.h5")):
            print(f"  missing J-lens data for step {step}")
            continue

        matched_steps.append(step)

        with (
            h5py.File(os.path.join(d, "stats.h5"), "r") as fs,
            h5py.File(os.path.join(d, "spectra.h5"), "r") as fsp,
        ):
            coh_vals, a_vals, tr_vals, alpha_vals, er_vals, spike_vals = (
                [], [], [], [], [], []
            )
            for li in range(12):
                key = f"layer_{li}"
                # stats.h5
                if key in fs:
                    attrs = dict(fs[key].attrs)
                    coh_vals.append(float(attrs.get("coherence", np.nan)))
                    a_vals.append(float(attrs.get("a_coeff", np.nan)))
                    tr_vals.append(float(attrs.get("tr_jtj_mean", np.nan)))
                else:
                    coh_vals.append(np.nan)
                    a_vals.append(np.nan)
                    tr_vals.append(np.nan)

                # spectra.h5
                if key in fsp and "eigenvalues" in fsp[key]:
                    ev = fsp[key]["eigenvalues"][:]
                    if ev is not None and len(ev) > 0:
                        ev_pos = np.maximum(ev, 1e-15)
                        ev_norm = ev_pos / ev_pos.sum()
                        entropy = -np.sum(ev_norm * np.log(ev_norm))
                        er_vals.append(np.exp(entropy))
                        # alpha from stored attribute (computed by spectra.fit_power_law_tail)
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
    print("Step 1: Downloading evaluation JSONs...")
    download_evals()
    print(f"  Done ({len(os.listdir('eval_data/zero_shot'))} files)")

    print("\nStep 2: Extracting benchmark scores...")
    eval_data = extract_scores()
    eval_steps = eval_data["steps"]
    scores = eval_data["scores"]
    print(f"  {len(eval_steps)} checkpoints, {len(KEY_BENCHMARKS)} benchmarks")

    print("\nStep 3: Cross-joining with J-lens metrics...")
    jlens_data = extract_jlens_metrics(eval_steps=eval_steps)
    matched = jlens_data["matched_steps"]
    metrics = jlens_data["metrics"]
    print(f"  {len(matched)} matched checkpoints")

    print("\nStep 4: Saving cross_join.npz...")
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
    np.savez("eval_data/cross_join.npz", **out)

    # Quick summary
    print("\nTop correlations (|ρ| > 0.80):")
    for mkey in sorted(metrics.keys()):
        mvals = np.array(metrics[mkey])
        for bm in KEY_BENCHMARKS:
            bvals = np.array([scores[bm].get(s, np.nan) for s in matched])
            mask = ~np.isnan(mvals) & ~np.isnan(bvals)
            if mask.sum() >= 5:
                r, _ = stats.spearmanr(mvals[mask], bvals[mask])
                if abs(r) > 0.80:
                    print(f"  ρ = {r:+.4f}  |  {mkey:25s}  ×  {bm}")

    print("\nDone. Run `python scripts/plot_benchmark_corr.py` for figures.")


if __name__ == "__main__":
    main()
