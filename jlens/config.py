"""Configuration for jlens pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for a checkpoint-ladder Jacobian lens pipeline."""

    # Model
    model_id: str = "EleutherAI/pythia-160m-deduped"
    model_dtype: str = "float16"  # torch dtype for model loading (bf16 has RoPE issues on B300)
    trust_remote_code: bool = True

    # Prompts
    prompt_dataset: str = "Salesforce/wikitext"
    prompt_config: str = "wikitext-2-raw-v1"
    prompt_split: str = "test"
    n_prompts: int = 1000
    max_seq_len: int = 256
    prompt_seed: int = 42

    # Overlap (cross-prompt eigenvector overlap)
    overlap_seed: int = 123
    overlap_n_prompts: int | None = None  # defaults to n_prompts

    # Checkpoints
    checkpoint_step_start: int = 1  # skip step0 (random init)
    checkpoint_step_end: int = 143000
    checkpoints: Optional[list[str]] = None  # explicit list overrides range

    # Jacobian
    compute_per_prompt: bool = True  # accumulate J̄ and Σ online
    layers: Optional[list[int]] = None  # None = all layers
    dim_batch: int = 8  # output dims per backward pass (higher = more VRAM)

    # Output
    results_dir: Path = Path("results")
    cache_dir: Optional[Path] = None  # model cache

    # SLURM
    slurm_gpus: int = 1
    slurm_cpus: int = 8
    slurm_mem: str = "64G"
    slurm_time: str = "02:00:00"


@dataclass
class CheckpointResult:
    """Results for a single checkpoint."""

    step: int
    n_layers: int
    d_model: int
    n_prompts: int

    # Per-layer quantities (each is list of length n_layers)
    jbar_gram: list  # J̄⊤J̄ matrix per layer
    eigenvalues: list  # sorted eigenvalues per layer
    coherence: list  # κ_ℓ per layer
    a_coeff: list  # a_ℓ = tr(J̄)/d
    r_norm: list  # ‖R‖_F / √d
    q_eff: list  # MP effective aspect ratio
    power_law_alpha: list  # low-λ power-law exponent
    n_spikes: list  # spike count above MP threshold
    effective_rank: list  # exp(entropy) rank
    tr_jtj_mean: list  # E[tr(J⊤J)] per layer
