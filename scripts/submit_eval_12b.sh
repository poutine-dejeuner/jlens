#!/bin/bash
# Submit evaluation jobs for all 12B checkpoints missing benchmark scores.
#
# Usage:
#   bash scripts/submit_eval_12b.sh        # Submit all missing
#   bash scripts/submit_eval_12b.sh --check  # Only check what's missing

set -e

# All 48 J-lens 12B steps
JLENS_STEPS=(1 2 4 8 16 32 64 128 256 512
             2000 5000 9000
             13000 17000 21000 24000 28000 32000 36000 40000 43000
             47000 51000 55000 59000 62000 66000 70000 74000 78000
             82000 85000 89000 93000 97000
             101000 104000 108000 112000 116000 120000 123000
             127000 131000 135000 139000 143000)

# Our 7 target benchmarks
BENCHMARKS=("arc_easy" "arc_challenge" "piqa" "winogrande" "sciq" "lambada_openai" "wsc")

cd /mnt/home/jacob/vl/repos/jlens

# Find missing steps that:
# 1. Have J-lens data (stats.h5 exists)
# 2. Do not already have eval scores for all 7 benchmarks
MISSING=()
HAVE=()

for step in "${JLENS_STEPS[@]}"; do
    STEPDIR="/data/jlens/pythia-12b-deduped/step$(printf '%04d' $step)"
    EVAL_FILE="eval_data/12b_zero_shot/step${step}.json"

    if [ ! -d "$STEPDIR" ]; then
        continue
    fi

    # Use Python to check if we have all 7 benchmark scores
    IS_COMPLETE=$(python3 -c "
import json, sys
try:
    with open('${EVAL_FILE}') as f:
        d = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print('NO')
    sys.exit(0)

results = d.get('results', {})
benchmarks = ['arc_easy','arc_challenge','piqa','winogrande','sciq','lambada_openai','wsc']
missing = 0
for bm in benchmarks:
    r = results.get(bm, {})
    acc = r.get('acc', r.get('acc,none'))
    acc_norm = r.get('acc_norm', r.get('acc_norm,none'))
    if acc is None and acc_norm is None:
        missing += 1

if missing == 0:
    print('YES')
else:
    print(f'PARTIAL({missing} missing)')
" 2>/dev/null || echo "NO")

    case "$IS_COMPLETE" in
        YES)
            HAVE+=("$step")
            ;;
        *)
            MISSING+=("$step")
            ;;
    esac
done

echo "Already have complete eval data: ${#HAVE[@]} steps"
echo "Missing eval data: ${#MISSING[@]} steps"

if [ "${#HAVE[@]}" -gt 0 ]; then
    echo "Have steps: ${HAVE[*]}"
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "Missing steps: ${MISSING[*]}"
fi

if [ "${#MISSING[@]}" -eq 0 ]; then
    echo "Nothing to submit."
    exit 0
fi

# Check mode
if [ "$1" = "--check" ]; then
    exit 0
fi

# Submit in batches of 4 (so we don't occupy too many GPUs at once)
BATCH_SIZE=4
TOTAL=${#MISSING[@]}
echo
echo "Submitting $TOTAL jobs in batches of $BATCH_SIZE..."
echo

for ((i=0; i<${#MISSING[@]}; i+=BATCH_SIZE)); do
    BATCH=("${MISSING[@]:i:BATCH_SIZE}")
    
    echo "--- Batch $(( i / BATCH_SIZE + 1 ))/$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE )) ---"
    for step in "${BATCH[@]}"; do
        echo "  Submitting step $step..."
        sbatch scripts/eval_12b_benchmark.sh "$step"
    done

    if [ $(( i + BATCH_SIZE )) -lt ${#MISSING[@]} ]; then
        echo "  Waiting 10s before next batch..."
        sleep 10
    fi
done

echo
echo "Done. Monitor with: squeue -u \$USER | grep jl-eval"
