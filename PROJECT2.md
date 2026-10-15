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
