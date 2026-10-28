# Jlens averaged over random stuff

## Goal
The Jlens spectral metrics (average eigenvalues, coherence, n-spikes, the power
law exponent of the tail, etc) have shown **striking**, almost impossibly
strong correlations with popular benchmarks over training steps (for pythia models for which training checkpoints were published). At the same time, the Jlens is not remarkably better than the logit
lens at the various tasks described in the jlens paper, so we are not convinced
this is such a good tool to study llms. I want to resolve this tension between
these 2 facts. We are going to re fit a jlens for all layers and all
checkpoints of pythia 12b with a different averaging set than the wikitext set;
instead we will average using a set of prompts made of completely random
strings of tokens. the number and length of each prompts should be comparable.

An even better outcome: explaining why the spectral jlens metrics correlate
strongly to benchmark scores over training steps.

## Conventions
- Keep this file updated with progress
- Never use the login node to do even light computations. always compute on the
  compute nodes using slurm.
- use test based development. if you need to add some code, make a test that
  tests the code you are about to write. code until it passes the tests.

## Progress

### Persistence Homology Analysis (Qwen3.6-27B)

Separate from the main Pythia-12B random-token experiment, we conducted a **persistence homology analysis of token embeddings** from Qwen/Qwen3.6-27B to understand the topological structure of embedding manifolds.

#### Key Findings

**1. Embedding manifolds are highly structured, much simpler than random baselines**
- Both singleton embeddings (n=42,158, each token seen once) and contextual embeddings (n=42,158, final hidden states for each token in the model's vocabulary over 1000 Wikitext prompts) have **far fewer topological features** than a random baseline of 5×225 points in 5120D.
- H1 z-scores: -25 to -28 relative to random (extremely structured)
- **Singleton H1 evolves dramatically across layers** (from ~35 in early layers to 50-90 in deep layers, with an anomalous jump to 118 at Layer 59)
- **Contextual H1 stays flatter (~50-70)** throughout all layers, suggesting context regularizes the embedding geometry

**2. Layer 59 anomaly:** Singleton H1 jumps from ~55 to 118, H2 from ~4 to 41.

**3. Cycle representative analysis (full Ripser, 42K tokens) reveals interpretable H1 cycles**

Of 31,777 H1 cycles found, most are short (72% have 2-3 vertices). Many high-persistence cycles are subword-form variants. But a significant minority are genuinely interpretable:

| Category | Persistence | Example Tokens |
|----------|------------|----------------|
| **Directions** | 0.146 | West → South → North → East (true semantic cycle!) |
| **Approximation** | 0.163 | than → least → almost → exactly → nearly → about → roughly |
| **Multiplicity** | 0.156 | Double → Single → Simple → Multiple → Dual |
| **Media networks** | 0.153 | NBC → CBS → BBC → CBC → ESPN |
| **Major/Minor** | 0.126 | Major → Minor (casing variants) |
| **Creation events** | 0.122 | created → Located → Founded → Introduced |
| **Entertainment** | 0.121 | Disney → Marvel → Nintendo → Batman → Anime |
| **Crimes** | 0.120 | crimes → murdered → murders → killings → rapes |
| **German articles** | 0.129 | Die → Der → Den → Das |
| **Seasons/Time** | 0.130 | Year → Quarter → Season |
| **First names** | 0.154 | John → David → Michael → James → 45 common names |
| **Companies** | 0.156 | Microsoft → Google → Apple → Amazon → 53 corporations |
| **Materials** | 0.156 | Box → Engine → glass → steel → Pool → Wall → 53 materials |
| **Countries** | 0.178 | China → Russia → Australia → Japan → France → 37 entities |

**4. Days-of-week and months are surprisingly weak**
- Days-of-week: low persistence (0.080)
- The cardinal directions cycle is the closest to a "true semantic cycle" where adjacent tokens correspond to adjacent real-world concepts.

**5. Targeted homology**: Pre-defined semantic categories probed with contextualized embeddings:
- Directions had 10 H1 cycles (most structure)
- Numbers had 21 H1 cycles (mostly subword forms)
- Days-of-week had only 2 H1 cycles

**Interpretation:** Most H1 cycles capture **clusters of semantically similar tokens** arranged in a circular pattern, rather than ordered sequential loops.

### 2025-10-01: Pipeline built, single-checkpoint test passed
- Random-token prompt generator: `jlens/prompts.py::generate_random_prompts()`
- CLI flag: `--random-tokens` on `jlens run`
- SLURM scripts: `scripts/run_full_12b_random.sh`, `scripts/run_single_12b_random.sh`
- Test job (step 143000, 10 prompts): completed in 634s, no errors
- Results dir: `/data/jlens/pythia-12b-deduped-random/`
- Early comparison (step 143000, L0):
  - Random: a_coeff=0.2425, coherence=0.4497, tr_jtj_mean=1.27e4
  - Wikitext: a_coeff=0.1730, coherence=0.3396, tr_jtj_mean=2.75e3
  → Random tokens produce higher identity-coefficient and coherence — 
    the Jacobian is closer to identity without structured input.
- 86/86 tests passing (6 new for random prompts)
