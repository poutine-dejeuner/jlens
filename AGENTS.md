# Slurm
User jacob is used by other people besides me. only use one of the compute
nodes at a time. Each node has 8 b300. Make full use.

# Code convention
test based coding. since the jobs are so long, we cant afford to catch bugs
during execution. we will quickly loose hundreds of hours of time debugging
like that.

# Script cleanup
Only documented scripts survive in scripts/. After a script has served its
one-shot purpose (tests, experiments, debugging), delete it. The keep list:
- plot_*.py: all plotting scripts
- eval_wmt.py, download_evals.py, prepare_eval_data.py: eval pipeline
- prompt_pythia.py: prompt generation
- run_slice.py: SLURM slice dispatcher
- sweep_*.sh: SLURM job scripts
- bench_upstream.sh, compare_estimators.sh, test_upstream.sh: reference
- run_full.sh, run_missing.sh: production sweep scripts

# Documentation
When making modifications to the code, update the skills files in ./.pi. When
Also keep these updated:
docs/BENCHMARKS.md
docs/PROGRESS.md

# Job monitoring
Whenever you launch a SLURM job or any long-running background process,
stay in the loop until it finishes. Do NOT drop the conversation or move on
to unrelated work. Poll periodically with squeue, tail the output/error
logs, and report progress. If a job fails, diagnose the error immediately
and propose a fix. When it succeeds, report the results.

# Compute
read ./docs/SLURM.md
