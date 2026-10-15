"""HDF5-based storage for checkpoint results."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _or_nan(x):
    """Return x or np.nan if None."""
    return x if x is not None else np.nan

def _or_nan_int(x):
    """Return x or -1 if None or NaN (int version)."""
    if x is None:
        return -1
    try:
        if np.isnan(x):
            return -1
    except (TypeError, ValueError):
        pass
    return x


def save_checkpoint_result(
    results_dir: Path,
    step: int,
    stats: dict,
    spectral: dict,
    scratch_dir: Path | None = None,
) -> Path:
    """Save checkpoint results to HDF5.

    Directory structure:
      results_dir/
        step0001/
          stats.h5       # Accumulated Jacobian statistics
          spectra.h5     # Spectral analysis results
          metadata.json  # Summary metadata

    Args:
        results_dir: Base results directory (may be NFS).
        step: Training step.
        stats: Accumulator output (jbar, jtj_gram, coherence, etc.).
        spectral: Spectral analysis output (eigenvalues, q_eff, etc.).
        scratch_dir: Optional local fast storage.  If set, HDF5 files
            are written here first then moved to results_dir for fast
            NFS writes.

    Returns:
        Path to the checkpoint results directory.
    """
    step_dir = results_dir / f"step{step:04d}"
    write_dir = (scratch_dir / f"step{step:04d}") if scratch_dir else step_dir
    write_dir.mkdir(parents=True, exist_ok=True)

    # Save stats
    with h5py.File(write_dir / "stats.h5", "w") as f:
        n_layers = len(stats["jbar"])
        f.attrs["n_prompts"] = stats["n"]
        f.attrs["n_layers"] = n_layers
        f.attrs["model_id"] = stats.get("model_id", "")

        for i in range(n_layers):
            grp = f.create_group(f"layer_{i}")
            jbar_i = stats["jbar"][i]
            jtj_gram_i = stats["jtj_gram"][i]
            sigma_gram_i = stats.get("sigma_gram", [None] * n_layers)[i]
            if jbar_i is not None:
                grp.create_dataset("jbar", data=jbar_i)
                grp.create_dataset("jtj_gram", data=jtj_gram_i)
            if sigma_gram_i is not None:
                grp.create_dataset("sigma_gram", data=sigma_gram_i)
            grp.attrs["coherence"] = stats["coherence"][i] if stats["coherence"][i] is not None else np.nan
            grp.attrs["a_coeff"] = stats["a_coeff"][i] if stats["a_coeff"][i] is not None else np.nan
            grp.attrs["r_norm"] = stats["r_norm"][i] if stats["r_norm"][i] is not None else np.nan
            grp.attrs["tr_jtj_mean"] = stats["tr_jtj_mean"][i] if stats["tr_jtj_mean"][i] is not None else np.nan
            grp.attrs["fcr"] = _or_nan(stats.get("fcr", [None] * n_layers)[i])

    # Save spectral results
    with h5py.File(write_dir / "spectra.h5", "w") as f:
        n_layers = spectral["n_layers"]
        f.attrs["n_layers"] = n_layers
        f.attrs["d_model"] = spectral["d_model"]
        f.attrs["n_prompts"] = spectral["n_prompts"]
        f.attrs["model_id"] = spectral.get("model_id", "")

        for i in range(n_layers):
            grp = f.create_group(f"layer_{i}")
            ev = spectral["eigenvalues"][i]
            if ev is not None:
                grp.create_dataset("eigenvalues", data=ev)
            sigma_ev = spectral.get("sigma_eigenvalues", [None] * n_layers)[i]
            if sigma_ev is not None and not np.isnan(sigma_ev).all():
                grp.create_dataset("sigma_eigenvalues", data=sigma_ev)
            grp.attrs["coherence"] = _or_nan(spectral["coherence"][i])
            grp.attrs["a_coeff"] = _or_nan(spectral["a_coeff"][i])
            grp.attrs["r_norm"] = _or_nan(spectral["r_norm"][i])
            grp.attrs["q_eff"] = _or_nan(spectral["q_eff"][i])
            grp.attrs["n_spikes"] = _or_nan_int(spectral["n_spikes"][i])
            grp.attrs["power_law_alpha"] = _or_nan(spectral["power_law_alpha"][i])
            grp.attrs["effective_rank"] = _or_nan(spectral["effective_rank"][i])
            grp.attrs["sigma_eff_rank"] = _or_nan(spectral.get("sigma_eff_rank", [np.nan] * n_layers)[i])
            grp.attrs["mp_sigma2"] = _or_nan(spectral["mp_sigma2"][i])
            grp.attrs["tr_jtj_mean"] = _or_nan(spectral["tr_jtj_mean"][i])
            grp.attrs["fcr"] = _or_nan(spectral.get("fcr", [np.nan] * n_layers)[i])
            grp.attrs["eigenvector_overlap"] = _or_nan(spectral.get("eigenvector_overlap", [np.nan] * n_layers)[i])

    # Save metadata
    metadata = {
        "step": step,
        "n_layers": spectral["n_layers"],
        "d_model": spectral["d_model"],
        "n_prompts": spectral["n_prompts"],
    }
    with open(write_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Move from scratch to results if using scratch
    if scratch_dir:
        import shutil
        step_dir.mkdir(parents=True, exist_ok=True)
        for name in ["stats.h5", "spectra.h5", "metadata.json"]:
            shutil.move(str(write_dir / name), str(step_dir / name))
        write_dir.rmdir()

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
        sigma_gram = []
        coherence = []
        a_coeff = []
        r_norm = []
        tr_jtj_mean = []
        fcr = []

        for i in range(n_layers):
            grp = f[f"layer_{i}"]
            # Skip layers that have no data (accumulator None-tolerant)
            if "jbar" in grp:
                jbar.append(np.array(grp["jbar"]))
                jtj_gram.append(np.array(grp["jtj_gram"]))
            else:
                jbar.append(None)
                jtj_gram.append(None)
            if "sigma_gram" in grp:
                sigma_gram.append(np.array(grp["sigma_gram"]))
            else:
                sigma_gram.append(None)
            coherence.append(grp.attrs.get("coherence", None))
            a_coeff.append(grp.attrs.get("a_coeff", None))
            r_norm.append(grp.attrs.get("r_norm", None))
            tr_jtj_mean.append(grp.attrs.get("tr_jtj_mean", None))
            fcr.append(grp.attrs.get("fcr", None))

        result["stats"] = {
            "model_id": f.attrs.get("model_id", ""),
            "n": n_prompts,
            "jbar": jbar,
            "jtj_gram": jtj_gram,
            "sigma_gram": sigma_gram,
            "coherence": coherence,
            "a_coeff": a_coeff,
            "r_norm": r_norm,
            "tr_jtj_mean": tr_jtj_mean,
            "fcr": fcr,
        }

    # Load spectral
    with h5py.File(step_dir / "spectra.h5", "r") as f:
        n_layers = f.attrs["n_layers"]

        eigenvalues = []
        sigma_eigenvalues = []
        spectral_coherence = []
        q_eff_list = []
        n_spikes_list = []
        power_law_alpha_list = []
        effective_rank_list = []
        fcr_list = []
        sigma_eff_rank_list = []
        overlap_list = []

        for i in range(n_layers):
            grp = f[f"layer_{i}"]
            if "eigenvalues" in grp:
                eigenvalues.append(np.array(grp["eigenvalues"]))
            else:
                eigenvalues.append(None)
            if "sigma_eigenvalues" in grp:
                sigma_eigenvalues.append(np.array(grp["sigma_eigenvalues"]))
            else:
                sigma_eigenvalues.append(None)
            spectral_coherence.append(grp.attrs.get("coherence", None))
            q_eff_list.append(grp.attrs.get("q_eff", None))
            n_spikes_list.append(grp.attrs.get("n_spikes", None))
            power_law_alpha_list.append(grp.attrs.get("power_law_alpha", None))
            effective_rank_list.append(grp.attrs.get("effective_rank", None))
            fcr_list.append(grp.attrs.get("fcr", None))
            sigma_eff_rank_list.append(grp.attrs.get("sigma_eff_rank", None))
            overlap_list.append(grp.attrs.get("eigenvector_overlap", None))

        result["spectral"] = {
            "n_layers": n_layers,
            "d_model": f.attrs["d_model"],
            "n_prompts": f.attrs["n_prompts"],
            "model_id": f.attrs.get("model_id", ""),
            "eigenvalues": eigenvalues,
            "sigma_eigenvalues": sigma_eigenvalues,
            "coherence": spectral_coherence,
            "q_eff": q_eff_list,
            "n_spikes": n_spikes_list,
            "power_law_alpha": power_law_alpha_list,
            "effective_rank": effective_rank_list,
            "fcr": fcr_list,
            "sigma_eff_rank": sigma_eff_rank_list,
            "eigenvector_overlap": overlap_list,
        }

    return result


def save_thought_manifold(
    results_dir: Path,
    step: int,
    manifold: dict,
    scratch_dir: Path | None = None,
) -> Path:
    """Save thought manifold results to HDF5.

    Files written to step_dir/manifold.h5:
        datasets: eigenvalues, eigenvectors, explained_variance_ratio,
                  singular_values, representations
        attrs: centered, V, d, model_id, step
    Also writes metadata.json alongside HDF5.

    Args:
        results_dir: Base results directory.
        step: Training step (or 0 for non-checkpointed models).
        manifold: Output from thought_manifold_pipeline / compute_similarity_eigenvectors.
        scratch_dir: Optional local fast storage for NFS efficiency.

    Returns:
        Path to the step directory.
    """
    step_dir = results_dir / f"step{step:04d}"
    write_dir = (scratch_dir / f"step{step:04d}") if scratch_dir else step_dir
    write_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(write_dir / "manifold.h5", "w") as f:
        f.create_dataset("eigenvalues", data=manifold["eigenvalues"])
        f.create_dataset("eigenvectors", data=manifold["eigenvectors"])
        f.create_dataset("explained_variance_ratio",
                         data=manifold["explained_variance_ratio"])
        f.create_dataset("singular_values", data=manifold["singular_values"])

        if "representations" in manifold:
            f.create_dataset("representations", data=manifold["representations"])

        f.attrs["centered"] = manifold.get("centered", True)
        f.attrs["V"] = manifold.get("V", 0)
        f.attrs["d"] = manifold.get("d", 0)
        f.attrs["model_id"] = manifold.get("model_id", "")
        f.attrs["step"] = step
        f.attrs["k"] = len(manifold["eigenvalues"])

    # Metadata
    metadata = {
        "step": step,
        "model_id": manifold.get("model_id", ""),
        "revision": manifold.get("revision", ""),
        "V": manifold.get("V", 0),
        "d": manifold.get("d", 0),
        "k": len(manifold["eigenvalues"]),
        "centered": manifold.get("centered", True),
    }
    with open(write_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if scratch_dir:
        import shutil
        step_dir.mkdir(parents=True, exist_ok=True)
        for name in ["manifold.h5", "metadata.json"]:
            shutil.move(str(write_dir / name), str(step_dir / name))
        write_dir.rmdir()

    logger.info(f"Saved thought manifold for step {step} to {step_dir}")
    return step_dir


def load_thought_manifold(
    results_dir: Path,
    step: int,
) -> dict | None:
    """Load thought manifold results from HDF5.

    Returns:
        Dict matching compute_similarity_eigenvectors output, or None if not found.
    """
    step_dir = results_dir / f"step{step:04d}"
    manifold_path = step_dir / "manifold.h5"

    if not manifold_path.exists():
        return None

    with h5py.File(manifold_path, "r") as f:
        result = {
            "eigenvalues": np.array(f["eigenvalues"]),
            "eigenvectors": np.array(f["eigenvectors"]),
            "explained_variance_ratio": np.array(f["explained_variance_ratio"]),
            "singular_values": np.array(f["singular_values"]),
            "centered": bool(f.attrs.get("centered", True)),
            "V": int(f.attrs.get("V", 0)),
            "d": int(f.attrs.get("d", 0)),
        }
        if "representations" in f:
            result["representations"] = np.array(f["representations"])

    return result


def is_thought_manifold_done(results_dir: Path | str, step: int) -> bool:
    """Check if thought manifold has been computed for a step."""
    results_dir = Path(results_dir)
    step_dir = results_dir / f"step{step:04d}"
    return (step_dir / "manifold.h5").exists()


def is_checkpoint_done(results_dir: Path | str, step: int, require_overlap: bool = False) -> bool:
    """Check if a checkpoint has already been processed.

    When require_overlap=True, also checks that the eigenvector_overlap
    attribute is present in the stored spectra.
    """
    results_dir = Path(results_dir)
    step_dir = results_dir / f"step{step:04d}"
    metadata_ok = (step_dir / "metadata.json").exists()
    if not metadata_ok:
        return False
    if require_overlap:
        spectra_path = step_dir / "spectra.h5"
        if not spectra_path.exists():
            return False
        try:
            import h5py
            with h5py.File(spectra_path, "r") as f:
                # Check first available layer for overlap key
                for key in sorted(f.keys()):
                    if "eigenvector_overlap" in f[key].attrs:
                        return True
                    break  # check only the first layer
            return False
        except Exception:
            return False
    return True
