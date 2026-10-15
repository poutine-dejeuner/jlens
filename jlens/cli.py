"""CLI for jlens pipeline."""

import logging
import sys
from pathlib import Path

import click

from .config import PipelineConfig
from .runner import run_pipeline

logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool):
    """jlens: Jacobian lens spectral analysis across pretraining checkpoints."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--model-id", default="EleutherAI/pythia-160m-deduped",
              help="HuggingFace model ID")
@click.option("--dtype", default="float16",
              type=click.Choice(["float16", "bfloat16", "float32"]),
              help="Model dtype (use float16 on B300; bfloat16 has RoPE issues)")
@click.option("--n-prompts", default=1000, type=int,
              help="Number of prompts per checkpoint")
@click.option("--max-seq-len", default=256, type=int,
              help="Maximum sequence length")
@click.option("--prompt-dataset", default="Salesforce/wikitext",
              help="Prompt dataset name")
@click.option("--prompt-seed", default=42, type=int,
              help="Random seed for prompt selection")
@click.option("--results-dir", default="results", type=click.Path(),
              help="Results directory")
@click.option("--cache-dir", default=None, type=click.Path(),
              help="Model cache directory")
@click.option("--start-step", default=1, type=int,
              help="First checkpoint step (skip step0)")
@click.option("--end-step", default=143000, type=int,
              help="Last checkpoint step")
@click.option("--layers", default=None, type=str,
              help="Comma-separated layer indices, or 'all'")
@click.option("--checkpoints", default=None, type=str,
              help="Comma-separated explicit checkpoint steps (overrides start/end)")
@click.option("--dim-batch", default=8, type=int,
              help="Output dims per backward pass (higher = more VRAM, same FLOPs)")
@click.option("--scratch-dir", default=None, type=click.Path(),
              help="Local scratch directory for HDF5 writes before NFS move")
@click.option("--overlap", is_flag=True, default=False,
              help="Enable cross-prompt eigenvector overlap measurement")
@click.option("--overlap-seed", default=123, type=int,
              help="Random seed for overlap prompt batch")
@click.option("--overlap-n-prompts", default=None, type=int,
              help="Number of overlap prompts (default: same as --n-prompts)")
@click.option("--random-tokens", is_flag=True, default=False,
              help="Use uniformly random token sequences instead of wikitext prompts")
def run(
    model_id: str,
    dtype: str,
    n_prompts: int,
    max_seq_len: int,
    prompt_dataset: str,
    random_tokens: bool,
    prompt_seed: int,
    results_dir: str,
    cache_dir: str | None,
    start_step: int,
    end_step: int,
    layers: str | None,
    checkpoints: str | None,
    dim_batch: int,
    scratch_dir: str | None,
    overlap: bool,
    overlap_seed: int,
    overlap_n_prompts: int | None,
):
    """Run the full checkpoint pipeline."""
    config = PipelineConfig(
        model_id=model_id,
        model_dtype=dtype,
        n_prompts=n_prompts,
        max_seq_len=max_seq_len,
        prompt_dataset=prompt_dataset,
        prompt_seed=prompt_seed,
        random_tokens=random_tokens,
        results_dir=Path(results_dir),
        cache_dir=Path(cache_dir) if cache_dir else None,
        checkpoint_step_start=start_step,
        checkpoint_step_end=end_step,
        dim_batch=dim_batch,
        scratch_dir=Path(scratch_dir) if scratch_dir else None,
    )

    if overlap:
        config.overlap_seed = overlap_seed
        config.overlap_n_prompts = overlap_n_prompts

    if layers and layers != "all":
        config.layers = [int(l.strip()) for l in layers.split(",")]

    if checkpoints:
        config.checkpoints = [c.strip() for c in checkpoints.split(",")]

    logger.info(f"Pipeline config: {config}")
    run_pipeline(config)


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def analyze(results_dir: str):
    """Analyze collected results."""
    from .spectra import analyze_checkpoint
    from .storage import load_checkpoint_result

    results_path = Path(results_dir)
    step_dirs = sorted(
        d for d in results_path.iterdir()
        if d.is_dir() and d.name.startswith("step")
    )

    click.echo(f"Found {len(step_dirs)} checkpoint results")

    for step_dir in step_dirs:
        step = int(step_dir.name.replace("step", ""))
        result = load_checkpoint_result(results_path, step)
        spectral = result["spectral"]

        click.echo(f"\nStep {step}:")
        click.echo(f"  Layers: {spectral['n_layers']}")
        for i in range(spectral["n_layers"]):
            click.echo(
                f"  L{i:02d}: κ={spectral['coherence'][i]:.3f} "
                f"a={result['stats']['a_coeff'][i]:.3f} "
                f"q_eff={spectral['q_eff'][i]:.2f} "
                f"spikes={spectral['n_spikes'][i]} "
                f"α={spectral['power_law_alpha'][i]:.3f} "
                f"eff_rank={spectral['effective_rank'][i]:.0f}"
            )


@main.command()
@click.option("--model-id", default="Qwen/Qwen3.6-27B",
              help="HuggingFace model ID")
@click.option("--revision", default=None, type=str,
              help="Model revision/branch (e.g., 'step1000' for Pythia checkpoints)")
@click.option("--step", default=None, type=int,
              help="Training step (convenience for revision=step<N>)")
@click.option("--dtype", default="bfloat16",
              type=click.Choice(["float16", "bfloat16", "float32"]),
              help="Model dtype (bfloat16 works for Qwen models)")
@click.option("--k", default=20, type=int,
              help="Number of top eigenvectors to compute")
@click.option("--batch-size", default=256, type=int,
              help="Forward pass batch size")
@click.option("--center/--no-center", default=True,
              help="Center embeddings before L2-normalization")
@click.option("--max-token-id", default=None, type=int,
              help="Max token ID (default: vocab_size)")
@click.option("--results-dir", default="results", type=click.Path(),
              help="Results directory")
@click.option("--cache-dir", default=None, type=click.Path(),
              help="Model cache directory")
@click.option("--layout", default=None, type=str,
              help="Dotted layout path (e.g., 'model.language_model' for multimodal)")
@click.option("--contextual/--singleton", default=False,
              help="Option B: contextualized averaging from Wikipedia EN "
                   "(default: Option A singleton)")
@click.option("--n-contexts", default=100, type=int,
              help="Target contexts per token (Option B only)")
def thought_manifold(
    model_id: str,
    revision: str | None,
    step: int | None,
    dtype: str,
    k: int,
    batch_size: int,
    center: bool,
    max_token_id: int | None,
    results_dir: str,
    cache_dir: str | None,
    layout: str | None,
    contextual: bool,
    n_contexts: int,
):
    """Compute thought manifold: token similarity eigenvectors.

    Two modes:
      - Option A (--singleton, default): singleton forward passes for
        every vocab token → Gram+eigh. Captures static vocabulary
        geometry (language families, subword morphology).
      - Option B (--contextual): contextualized averaging from Wikipedia
        EN → Gram+eigh. Captures semantic structure (part-of-speech,
        animacy, concreteness).

    Uses the final-layer residual stream (post-norm, pre-LM-head).
    """
    from .thought_manifold import run_thought_manifold_for_checkpoint
    from .storage import save_thought_manifold

    results_path = Path(results_dir)
    actual_step = step if step is not None else 0

    if revision is None and step is not None:
        revision = f"step{step}"

    result = run_thought_manifold_for_checkpoint(
        model_id=model_id,
        step=actual_step,
        revision=revision,
        k=k,
        batch_size=batch_size,
        center=center,
        dtype=dtype,
        cache_dir=str(cache_dir) if cache_dir else None,
        layout_str=layout,
        contextual=contextual,
        n_contexts=n_contexts,
    )

    save_thought_manifold(results_path, actual_step, result)

    click.echo(f"\nModel: {model_id}")
    click.echo(f"Vocab: {result['V']} tokens, d_model: {result['d']}")
    click.echo(f"Centered: {result['centered']}")
    click.echo(f"\nTop-{k} eigenvalues:")
    for i, ev in enumerate(result["eigenvalues"]):
        evr = result["explained_variance_ratio"][i]
        click.echo(f"  λ_{i+1}: {ev:.4f}  ({100*evr:.2f}% variance)")
    click.echo(f"\nCumulative explained variance: "
               f"{100*result['explained_variance_ratio'].sum():.2f}%")
    click.echo(f"\nSaved to {results_path}")


@main.command()
def version():
    """Print version."""
    from . import __version__
    click.echo(f"jlens {__version__}")


if __name__ == "__main__":
    main()
