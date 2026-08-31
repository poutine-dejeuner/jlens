# Direction IV: Developmental Emergence of the Jacobian Lens

## Goal

Track the spectral evolution of the Jacobian lens across Pythia-160M pretraining
checkpoints. Map how workspace signatures (coherence, spectral shape, spike
population, power-law tails) emerge during training, and correlate with known
developmental milestones in the model.

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | Pythia-160M-deduped | 154 checkpoints, d=768, L=12. Smallest model for fast iteration. |
| Checkpoints | step1–step143000 (skip step0) | step0 is random init, J ≈ I, uninteresting |
| Prompts | 1000 from wikitext | Matches the companion notebook's setup |
| Jacobian object | ∂h_final/∂h_ℓ (d×d) | Spectrally equivalent to W_U J_ℓ; no need for vocab-dim readout |
| Storage target | J⊤J per prompt, not full J | J⊤J gives spectra + coherence via tr(J⊤J). Cheaper to compute. |
| Compute approach | per-prompt vjp → accumulate J̄, Σ (online) | Full per-prompt Jacobian Gram matrices enable coherence κ_ℓ |
| Architecture | Python package with CLI + resume | Robust for 154-checkpoint batch job |

## Why J⊤J instead of full J

The spectral analysis works on eigenvalues of J⊤J. We need:

- **Spectrum of J̄⊤J̄**: eigenvalues of the averaged Gram matrix. Compute running
  J̄ online, then form J̄⊤J̄ and eigendecompose once per checkpoint.
- **Coherence κ_ℓ**: κ = tr(J̄⊤J̄) / tr(E[J⊤J]). Need tr(J⊤J) per prompt.
  Compute tr(J⊤J(x)) = ‖J(x)‖_F² per prompt via vjp trick:
  for each output dimension j, compute ‖∂(h_final)_j/∂h_ℓ‖², sum over j.
  This is d vjp calls per prompt per layer (d=768).
  **Optimization**: use Hutchinson trace estimator — for random v ∼ N(0,I),
  E[v⊤ J⊤J v] = tr(J⊤J). A few random probes per prompt give unbiased estimate
  of tr(J⊤J) with variance ∝ 1/k, much cheaper than d vjps.
- **Eigenvectors of J̄⊤J̄**: for spike overlap analysis. Available from the
  averaged Gram eigendecomposition.

**Computational note**: The Hutchinson estimator gives us tr(J⊤J) per prompt
(needed for κ_ℓ) without materializing the full d×d Jacobian. For the averaged
J̄ itself, we compute vjp per output dimension and average — this gives us J̄
directly (d² storage). This is the exact same cost as the paper's method.

## Checkpoint structure

From HuggingFace: `EleutherAI/pythia-160m-deduped`

- 10 log-spaced early: step1, step2, step4, step8, step16, step32, step64, step128, step256, step512
- 143 evenly-spaced: step1000 through step143000 (every 1000 steps)
- step143000 = main branch (final model)
- Total to process: 153 checkpoints (skip step0)
- Each checkpoint: ~625MB on disk (fp32), ~325MB (fp16)

## Package architecture

```
jspace/
├── pyproject.toml
├── plan.md
├── jlens/
│   ├── __init__.py
│   ├── cli.py                  # click CLI: run, analyze, plot
│   ├── config.py               # dataclass config, YAML loading
│   ├── checkpoint_manager.py   # discover, load, cache Pythia checkpoints
│   ├── prompts.py              # wikitext loader, tokenization
│   ├── jacobian.py             # core: per-prompt vjp, accumulate J̄, Σ
│   ├── accumulator.py          # online Welford-style accumulation of J̄, Σ, κ
│   ├── spectra.py              # SVD, MP fit, power-law fit, spike count
│   ├── runner.py               # checkpoint iteration with resume
│   ├── storage.py              # HDF5 save/load of per-checkpoint results
│   └── plotting.py             # heatmaps, depth×step curves
├── configs/
│   └── pythia-160m.yaml
├── scripts/
│   └── run_pipeline.sh
└── results/                    # output directory
```

## Per-checkpoint pipeline

```
For each checkpoint (step s):
  1. Load model in bf16 on GPU, set eval mode
  2. Hook residual stream at each layer ℓ ∈ {0..L-1}
  3. For each of N=1000 prompts x_i:
     a. Tokenize, forward pass
     b. For each layer ℓ, compute ∂(h_final)/∂h_ℓ via autodiff
        - This gives J_ℓ(x_i) as a d×d matrix
     c. Accumulate online:
        J̄_ℓ  ← running mean of J_ℓ(x_i)
        Σ_ℓ   ← running sum of (J - J̄)⊤(J - J̄)
        tr_E  ← running mean of tr(J⊤J) = ‖J‖²_F
  4. After all prompts:
     a. J̄_ℓ = accumulated mean
     b. J̄_ℓ⊤J̄_ℓ, eigendecompose → eigenvalues λ_k
     c. κ_ℓ = tr(J̄⊤J̄) / tr_E
     d. a_ℓ = tr(J̄)/d, ‖R‖_F/√d = sqrt(tr((J̄ - aI)⊤(J̄ - aI))/d)
     e. MP fit → q_eff(ℓ)
     f. Power-law fit on low-λ tail → exponent α
     g. Spike count above MP threshold
     h. Effective rank = exp(H), H = -Σ p_k log p_k, p_k = λ_k/Σλ
  5. Save results/<step>.npz with all above
  6. Unload model, clear CUDA cache
```

## Key quantities per (layer, step)

| Quantity | Symbol | What it measures |
|---|---|---|
| Eigenvalues | λ_k(J̄⊤J̄) | Full spectrum per layer |
| Coherence fraction | κ_ℓ | How context-independent transport is |
| Identity coefficient | a_ℓ = tr(J̄)/d | How close J̄ is to aI |
| Off-identity norm | ‖R‖_F/√d | How much non-identity structure exists |
| MP effective aspect ratio | q_eff | d / effective rank of MP bulk |
| Power-law tail exponent | α | Inter-layer alignment: α → 0.45 = aligned |
| Spike count | n_spikes | # eigenvalues above MP threshold |
| Effective rank | exp(H(λ)) | Entropy-based dimensionality |

## Predicted patterns to test

From the paper (Direction IV predictions):

1. **CKA block structure absent early** → three blocks crystallize
2. **Workspace band crystallizes** near 128M-4B token window
   - Pythia-160M: 128M tokens ≈ step61, 4B tokens ≈ step1907
   - Our checkpoints: step64, step1000 bracket this window
3. **Power-law exponent α**:
   - Starts near Fuss-Catalan value ~1/(N+1) ≈ 0.08 (uncorrelated factors)
   - Falls toward trained value ≈ 0.45 as alignment develops
4. **Coherence κ_ℓ profile**: sharpens from flat → sigmoidal vs depth
5. **q_eff drainage**: shallow layers q_eff ≫ 1 early, workspace q_eff → 1
6. **a_ℓ → 1, ‖R‖_F → 0**: J approaches identity in deeper layers as
   residual updates become small (perturbative regime)

## Output visualizations

1. **Spectral evolution heatmap**: eigenvalues vs depth vs training step
   (3-panel: color = log density)
2. **κ_ℓ depth curves**: one line per checkpoint phase (early/mid/late),
   showing sharpening from flat to sigmoidal
3. **q_eff vs depth × step**: 2D heatmap, overlay q_eff=1 contour
4. **Power-law exponent vs training step**: per-layer lines, showing
   convergence toward ~0.45
5. **Spike count vs depth × step**: 2D heatmap
6. **a_ℓ and ‖R‖_F vs depth × step**: 2-panel heatmap
7. **Effective rank vs depth × step**: 2D heatmap
8. **Workspace onset detection**: q_eff crossover layer vs training step
   (how the "workspace band" emerges)

## Compute estimate

- Model: Pythia-160M, d=768, L=12
- 1000 prompts × ~128 tokens/prompt = 128K tokens/checkpoint
- Per prompt: 1 forward + L×d vjps = 1 fwd + 12×768 bwds ≈ 1 fwd + 9216 bwds
- Actually: we can compute ∂h_final/∂h_ℓ for all ℓ in one backward pass
  by back-propagating through the chain. Need to verify.
- 153 checkpoints
- Rough estimate: 153 × (1000 × 1 fwd+backward through 12 layers)
  ≈ 153K forward+backward passes through a 160M model
- On an A100: ~1-2 hours per checkpoint → 150-300 hours total
  (this seems high; need to profile)
- **Optimization opportunity**: if we accept Hutchinson trace for κ_ℓ,
  we can skip per-prompt J storage and just accumulate. Still need J̄.

## Dependencies

```
torch >= 2.0
transformers >= 4.30
datasets  # wikitext
h5py      # HDF5 storage
numpy, scipy, matplotlib, seaborn
click     # CLI
pyyaml    # config
tqdm      # progress
```
