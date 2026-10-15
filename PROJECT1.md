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
