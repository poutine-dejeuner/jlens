"""HDF5-based storage for checkpoint results."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def save_checkpoint_result(
    results_dir: Path,
    step: int,
    stats: dict,
    spectral: dict,
) -> Path:
    """Save checkpoint results to HDF5.

    Directory structure:
      results_dir/
        step0001/
          stats.h5       # Accumulated Jacobian statistics
          spectra.h5     # Spectral analysis results
          metadata.json  # Summary metadata

    Args:
        results_dir: Base results directory
        step: Training step
        stats: Accumulator output (jbar, jtj_gram, coherence, etc.)
        spectral: Spectral analysis output (eigenvalues, q_eff, etc.)

    Returns:
        Path to the checkpoint results directory.
    """
    step_dir = results_dir / f"step{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Save stats
    with h5py.File(step_dir / "stats.h5", "w") as f:
        n_layers = stats["n"][1] if isinstance(stats["n"], tuple) else len(stats["jbar"])
        f.attrs["n_prompts"] = stats["n"]
        f.attrs["n_layers"] = n_layers

        for i in range(n_layers):
            grp = f.create_group(f"layer_{i}")
            grp.create_dataset("jbar", data=stats["jbar"][i])
            grp.create_dataset("jtj_gram", data=stats["jtj_gram"][i])
            grp.attrs["coherence"] = stats["coherence"][i]
            grp.attrs["a_coeff"] = stats["a_coeff"][i]
            grp.attrs["r_norm"] = stats["r_norm"][i]
            grp.attrs["tr_jtj_mean"] = stats["tr_jtj_mean"][i]

    # Save spectral results
    with h5py.File(step_dir / "spectra.h5", "w") as f:
        n_layers = spectral["n_layers"]
        f.attrs["n_layers"] = n_layers
        f.attrs["d_model"] = spectral["d_model"]
        f.attrs["n_prompts"] = spectral["n_prompts"]

        for i in range(n_layers):
            grp = f.create_group(f"layer_{i}")
            grp.create_dataset("eigenvalues", data=spectral["eigenvalues"][i])
            grp.attrs["coherence"] = spectral["coherence"][i]
            grp.attrs["a_coeff"] = spectral["a_coeff"][i]
            grp.attrs["r_norm"] = spectral["r_norm"][i]
            grp.attrs["q_eff"] = spectral["q_eff"][i]
            grp.attrs["n_spikes"] = spectral["n_spikes"][i]
            grp.attrs["power_law_alpha"] = spectral["power_law_alpha"][i]
            grp.attrs["effective_rank"] = spectral["effective_rank"][i]
            grp.attrs["mp_sigma2"] = spectral["mp_sigma2"][i]
            grp.attrs["tr_jtj_mean"] = spectral["tr_jtj_mean"][i]

    # Save metadata
    metadata = {
        "step": step,
        "n_layers": spectral["n_layers"],
        "d_model": spectral["d_model"],
        "n_prompts": spectral["n_prompts"],
    }
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved results for step {step} to {step_dir}")
    return step_dir


def load_checkpoint_result(
    results_dir: Path,
    step: int,
) -> dict:
    """Load checkpoint results from HDF5.

    Returns:
        Dict with 'stats' and 'spectral' keys.
    """
    step_dir = results_dir / f"step{step:04d}"

    result = {}

    # Load stats
    with h5py.File(step_dir / "stats.h5", "r") as f:
        n_layers = f.attrs["n_layers"]
        n_prompts = f.attrs["n_prompts"]

        jbar = []
        jtj_gram = []
        coherence = []
        a_coeff = []
        r_norm = []
        tr_jtj_mean = []

        for i in range(n_layers):
            grp = f[f"layer_{i}"]
            jbar.append(np.array(grp["jbar"]))
            jtj_gram.append(np.array(grp["jtj_gram"]))
            coherence.append(grp.attrs["coherence"])
            a_coeff.append(grp.attrs["a_coeff"])
            r_norm.append(grp.attrs["r_norm"])
            tr_jtj_mean.append(grp.attrs["tr_jtj_mean"])

        result["stats"] = {
            "n": n_prompts,
            "jbar": jbar,
            "jtj_gram": jtj_gram,
            "coherence": coherence,
            "a_coeff": a_coeff,
            "r_norm": r_norm,
            "tr_jtj_mean": tr_jtj_mean,
        }

    # Load spectral
    with h5py.File(step_dir / "spectra.h5", "r") as f:
        n_layers = f.attrs["n_layers"]

        eigenvalues = []
        spectral_coherence = []
        q_eff_list = []
        n_spikes_list = []
        power_law_alpha_list = []
        effective_rank_list = []

        for i in range(n_layers):
            grp = f[f"layer_{i}"]
            eigenvalues.append(np.array(grp["eigenvalues"]))
            spectral_coherence.append(grp.attrs["coherence"])
            q_eff_list.append(grp.attrs["q_eff"])
            n_spikes_list.append(grp.attrs["n_spikes"])
            power_law_alpha_list.append(grp.attrs["power_law_alpha"])
            effective_rank_list.append(grp.attrs["effective_rank"])

        result["spectral"] = {
            "n_layers": n_layers,
            "d_model": f.attrs["d_model"],
            "n_prompts": f.attrs["n_prompts"],
            "eigenvalues": eigenvalues,
            "coherence": spectral_coherence,
            "q_eff": q_eff_list,
            "n_spikes": n_spikes_list,
            "power_law_alpha": power_law_alpha_list,
            "effective_rank": effective_rank_list,
        }

    return result


def is_checkpoint_done(results_dir: Path, step: int) -> bool:
    """Check if a checkpoint has already been processed."""
    step_dir = results_dir / f"step{step:04d}"
    return (step_dir / "metadata.json").exists()
