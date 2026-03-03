#!/bin/bash

# ─── Default values ──────────────────────────────────────────────────

EPOCHS=400
FILENAME="oitaven"

GPUS=1
BATCH=128
BATCH_GPU=64

SEED=42

# ─── Setup Logs & Paths ──────────────────────────────────────────────

OUT_DIR="job_outputs"
mkdir -p "$OUT_DIR"

UUID=$(cat /proc/sys/kernel/random/uuid | cut -c1-8)

LOG_FILE="${OUT_DIR}/${FILENAME}${SEED}_e${EPOCHS}_b${BATCH}_${UUID}.log"

SIF_IMAGE="$HOME/stylegan3.sif"
WORK_DIR=$(pwd)

# ─── Command ──────────────────────────────────────────────────────────

CMD="apptainer exec --nv --bind ${WORK_DIR}:/workspace --pwd /workspace ${SIF_IMAGE} \
bash scripts/run_training_classification.sh \
--cfg=stylegan3-t \
--epochs=${EPOCHS} \
--filename=${FILENAME} \
--seed=${SEED} \
--gpus=${GPUS} \
--batch=${BATCH} \
--eval-classification=True"

# ─── Execution ──────────────────────────────────────────────────────
echo "Epochs: $EPOCHS | Filename: $FILENAME | Seed: $SEED | GPUs: $GPUS | Batch Size: $BATCH | Batch Size per GPU: $BATCH_GPU"
echo "Log file: $LOG_FILE"

export PYTHONUNBUFFERED=1

JOB_ID=$(ts -G $GPUS bash -c "$CMD > $LOG_FILE 2>&1")

echo "job running ($JOB_ID)! Check log file for progress: $LOG_FILE"