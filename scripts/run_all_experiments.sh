#!/bin/bash

# ─── Experiment matrix ────────────────────────────────────────────────
#
# Each dataset is submitted as independent jobs to task-spooler (ts).
# Since each job requests 1 GPU (-G 1), ts will run as many in parallel
# as there are free GPU slots, spreading load across all available GPUs.
# Seeds within the same dataset are also independent and submitted
# concurrently — ts handles the scheduling.
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --epochs=200 --batch=64

DATASETS=(oitaven eiras ermidas mera ferreiras xesta ulla mestas)
SEEDS=(42 43 44 45 46)

# ─── Forward any extra args (e.g. --epochs, --batch) to run_job.sh ───
EXTRA_ARGS="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds = $(( ${#DATASETS[@]} * ${#SEEDS[@]} )) jobs"
echo ""

for FILENAME in "${DATASETS[@]}"; do
  if [[ "$FILENAME" == "ulla" ]]; then
    EPOCHS=800
  else
    EPOCHS=400
  fi

  for SEED in "${SEEDS[@]}"; do
    bash "${SCRIPT_DIR}/run_job.sh" \
      --filename="${FILENAME}" \
      --seed="${SEED}" \
      --epochs="${EPOCHS}" \
      ${EXTRA_ARGS}
  done
done

echo ""
echo "All jobs submitted. Monitor the queue with: ts"
echo "Watch progress with:  watch -n 5 ts"
