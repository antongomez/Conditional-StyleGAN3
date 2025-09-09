#!/bin/bash
#SBATCH --job-name=stylegan3-training       # Job name
#SBATCH --output=job_outputs/out_%j.out     # %j it will be replaced by the job ID
#SBATCH --error=job_outputs/error_%j.err
#SBATCH --ntasks=1                          # Number of tasks (usually 1 for single GPU jobs)
#SBATCH --cpus-per-task=64                  # Cores per task
#SBATCH --mem-per-cpu=3G                    # Memory per CPU core
#SBATCH --time=48:00:00                     # Maximum execution time (48 hours - maximum time for short queue / 3 days - maximum time for medium queue / 7 days - maximum time for long queue)
#SBATCH --gres=gpu:a100:2                   # Request 2 GPUs

# ─── Default values ──────────────────────────────────────────────────
EPOCHS=500
UNIFORM_CLASS=False
DISC_ON_GEN=True
GPUS=2
BATCH=64
OTHER_ARGS=""

FILENAME="oitaven"

# ─── Parse arguments ─────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --epochs=*)
      EPOCHS="${arg#*=}"
      ;;
     --filename=*)
      FILENAME="${arg#*=}"
      ;;
    --uniform-class=*)
      UNIFORM_CLASS="${arg#*=}"
      ;;
    --disc-on-gen=*)
      DISC_ON_GEN="${arg#*=}"
      ;;
    --gpus=*)
      GPUS="${arg#*=}"
      ;;
    --batch=*)
      BATCH="${arg#*=}"
      ;;
    *)
      OTHER_ARGS+="$arg "
      ;;
  esac
done


# ─── Build dataset names ──────────────────────────────────────────────
DATASET="${FILENAME}_train"
DATASET_VAL="${FILENAME}_val"

# ─── Load environment and run training ───────────────────────────────
module load singularity/3.9.7

echo "Starting Singularity container for StyleGAN3 training..."

singularity exec --nv --bind $(pwd):/workspace stylegan3.sif bash -c \
"bash run_training.sh \
  --dataset=$DATASET \
  --dataset-val=$DATASET_VAL \
  --epochs=$EPOCHS \
  --gpus=$GPUS \
  --cls-weight=0.1 \
  --batch=$BATCH \
  --uniform-class=$UNIFORM_CLASS \
  --disc-on-gen=$DISC_ON_GEN \
  $OTHER_ARGS"

echo "Done! Job finished successfully."

# Oitaven dataset example epochs and kimg calculations
# Oitaven --- 3020 img : 350 epochs ---> 1.080 Mimg
# Oitaven --- 3020 img : 500 epochs ---> 1.510 Mimg
# Oitaven --- 3020 img : 600 epochs ---> 1.812 Mimg