# Persistence homology cycles analysis

## Goal
Interpret (ie give a meaining to) the cycles/coclycles computed on both
datasets of embeddings: singleton and contextual.
The most striking signal would be to observe a cycle that encodes a well known
cyclicicity of concepts: days of the week, months, colors,
north-west-south-east etc.

## Remarks
There are very many versions of this experiment we should test.
- for both embeddings silngleton and contextual, there is a set of embeddings
  per layers. This is a lot, we should do the initial tests only for the last
layer.
- in a transformer, before going into the attention heads, the embeddings are
  normalized in some way (usual normalisation or rms-norm). We should also
consider the cycles of the normalized embeddings. The results will be
different.
- There are a lot of 1-cycles, even in homology. We need a subsampling
  strategy. I assume the longest living cycles are the most interesting ones,
(might not be fully true). Lets order the 1-cycles by life length, sample
cycles from different lengths to interpret. Is it the case that longer living
cycles are more readily interpretable?

## Conventions
- Keep this file updated with progress
- **NEVER** use the login node to do even light computations. always compute on the
  compute nodes using slurm.
- use test based development. if you need to add some code, make a test that
  tests the code you are about to write. code until it passes the tests.

## Major Finding: Persistent Homology Discovers Semantic Cycles

**Full 5120D Ripser on 42K contextual tokens (no threshold, maxdim=1) found
31,777 H1 cycles. Among them, automatic semantic concept groups were discovered
without any prior knowledge:**

| Concept | Best Cycle | Members Found | NN Adj Correct |
|---------|-----------|---------------|----------------|
| **Months** | 13 tokens | 12/12 canonical + "-may" | 7/13 |
| **Days** | 5 tokens | Fri, Mon, Thu, Sat, Sun | 2/5 |
| **Seasons** | 3 tokens | Spring, Autumn, Winter | 2/3 |
| **Directions** | 5 tokens | N, S, E, W (two "west") | 2/5 |

### Months: Cleanest Discovery
The 12 canonical months (January through December) plus the subword "-may"
appear as a single H1 cycle (birth=0.84, persistence=0.06). The distance
matrix reveals sub-clusters:
- {march, april, may}: 0.287-0.387
- {june, july}: 0.253
- {september, october, november, december}: 0.263-0.288
- {january, february}: 0.283
These sub-clusters bridge to each other at the cycle's birth radius (~0.84),
forming a topological 1-cycle.

### Days: Missing Tuesday/Wednesday
5 of 7 days found in one cycle (birth=0.79). Tuesday and Wednesday are absent
— their contextual embeddings (averaged over Wikipedia sentences) are pulled
away from the day-cycle geometry. The NN tour recovers the canonical ordering
partially (fri→thu correct, sat before sun correct).

### Data & Key Files
- **Full persistence diagram + 31,777 cocycles**:
  `/data/jlens/qwen3.6-27b/ripser_h1_full_nothresh.pkl` (3.9 MB)
- **Token-mapped cycles**:
  `/data/jlens/qwen3.6-27b/cycle_tokens_full.pkl` (5.0 MB, 31,777 entries)
- **Contextual token manifest**:
  `/data/jlens/qwen3.6-27b/contextual_token_manifest.json`
- **Contextual raw reps**: `/data/jlens/qwen3.6-27b/step0000/manifold.h5`

### Cycle Statistics
- Mean: 5.5 vertices/cycle, 4.9 edges/cycle
- 6,624 cycles with ≥5 vertices, 2,680 with ≥10 vertices
- Max: 1,617 vertices (morphological group: common English suffixes)
- Birth distribution: min=0.18, max=1.29, median=0.63
- Cycles with low birth (<0.5) are character-level duplicates (Latin/Cyrillic)
- Cycles with birth 0.6-0.9 contain semantic concept groups

### Large Cycles = Morphological Groups
The largest cycles (hundreds of vertices) are linguistic patterns: common
suffixes (-tion, -ing, -ers, -ment), single characters (a, e, s, t), and
subword fragments. These form connected components at moderate birth radii
due to distributional similarity in English text.

### Key Insight for Singleton
The contextual embeddings are *averages* over multiple Wikipedia contexts.
This explains why the months cycle has 7/13 correct adjacencies, not 12/13.
We hypothesize singleton (single-token forward passes) will produce cleaner
cycle geometry with higher adjacency scores.

## Progress

### Current approach: full-dataset persistent homology

### Completed
- [x] Full 5120D Ripser on 42K contextual: 31,777 H1 cycles (108 min)
- [x] Cocycle extraction: 31,777 representative edge sets
- [x] Token mapping: all cycles decoded to token strings
- [x] Semantic cycle discovery: months/days/seasons/directions found
- [x] Nearest-neighbor ordering analysis on discovered cycles

### Blocked on
- Singleton raw embeddings: need GPU node to compute 248K single-token forward passes
- Compare singleton cycles to contextual cycles (expect cleaner signal)

## Re-assessment of the PH cycles (2026-10-28)

The H1 "concept cycles" above are best read as **clusters, not cyclic
orderings**: persistence 0.06 at birth 0.84 (death/birth ≈ 1.07) is
noise-scale; 31.8K H1 classes on 42K points is what a noisy cloud gives; a
short-lived class's representative cycle is non-unique, so its vertex set
reflects a neighbourhood, not an order. A 12-point circle cannot survive in a
42K-token global Rips complex (other tokens fill the hole), and the global
top-10 PCA keeps frequency/morphology directions, not concept subspaces.
Also: contextual reps came from `last_hidden_state`, i.e. *after* the final
norm, the most output-shaped representation available.

## Circle probe: known cyclic concepts, all layers, per head

Code `jlens/circle_probe.py` (tests `tests/test_circle_probe.py`, 40 tests),
job `scripts/sweep_circle_probe.sh`, plots `scripts/plot_circle_probe.py`.
Data `/data/jlens/qwen3.6-27b/circle_probe/{probe.h5,scores.csv}`.

Qwen3.6-27B is hybrid: 64 layers, every 4th (3, 7, …, 63) is full gated
softmax attention (24 q heads, 4 kv heads, head_dim 256); the others are
Gated DeltaNet (48 value heads × 128).

Setup: 8 concepts × 16 templates (word never at position 0), rep at the
word's last subword token. Cyclic: days(7), months(12), seasons(4),
directions(8), hours("{one..twelve} o'clock"), colors(6 hues). Controls:
numbers (same words one..twelve in *counting* contexts → open line) and
12 arbitrary nouns. Units (2,881 per concept): residual stream (65),
normed residual = mixer input (64), value heads (16×4), attention head
outputs pre-o_proj gated (16×24), DeltaNet head outputs pre-out_proj (48×48).
Template mean removed, word centroids in top-32 PCs, 200 label permutations.

### Lesson: PCA "circles" and LOO angle fits are not evidence of a loop
`pca_order` (angle order in top-2 PCs) and `loo` (leave-one-word-out ridge
fit of cos/sin) were high everywhere, *including the numbers control*
(90–100% of units significant). Any smooth, curved, open 1-D path gives a
PCA horseshoe and is interpolable from neighbours. Only the **wrap** tests
closure: `circ_minus_lin` = Spearman(dist, circular index dist) −
Spearman(dist, linear index dist) (>0 loop, <0 open) and `closure` =
d(last, first) / mean step. Synthetic power: detects a circle hidden in a
2-D subspace under per-word noise up to ~2× its radius.

### Result: ordered, but open — no loops above chance
| concept | ordinal units (loo) | loop units (circ>lin, p<.01, closure<2) | median circ−lin | median closure |
|---|---|---|---|---|
| days | 57–84% | 0–1.0% | −0.19 | 1.49 |
| months | 85–98% | 0–1.6% | −0.08 | 1.35 |
| hours | 95–100% | 0–1.3% | −0.23 | 2.33 |
| numbers (ctrl) | 90–100% | 0–0.5% | −0.29 | 2.09 |
| nouns (ctrl) | 0% | 0–1.5% | +0.03 | 0.98 |

Loop-unit rates for real cyclic concepts equal the nouns control / the
1% false-positive rate, at every depth and in every head type. Earlier
layers and individual heads do not reveal hidden cycles. Hours ≈ numbers:
"twelve" stays far from "one" even in clock contexts. Seasons (4 items)
is underpowered (min p ≈ 0.07). Supervised closure (`loo_circ_minus_lin`)
agrees.

### One lead (needs replication)
Layer-55 attention heads h4 and h0 close a clock loop for hours
(circ−lin 0.42 / 0.34, closure ≈ 1.0, see `figures/circle_probe_top_pca.png`)
but stay open for the *same words* in counting contexts (−0.19 / −0.31,
closure ~1.5): context-specific. h9 / h8 loop in both (lexical).
Selected from 2,881 units → must replicate on held-out templates.

### Next
- Split-half template replication of every loop unit (CPU only).
- Fresh templates / Wikipedia contexts for hours, layer-55 heads.
- If L55 h4 replicates: activation patching along its circle (does rotating
  the angle shift "three o'clock" → "four o'clock" predictions?).
