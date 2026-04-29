#!/bin/bash

# ==============================
# DEBUG + SAFETY
# ==============================
set -eo pipefail
# set -x  # Uncomment this line only if you need to debug line-by-line

# ==============================
# RESOLVE PATHS
# ==============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "CURRENT DIR: $(pwd)"

# ==============================
# ENV DETECTION
# ==============================
IS_SLURM=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  IS_SLURM=1
fi

JOB_ID=${SLURM_JOB_ID:-LOCAL}
JOB_NAME=${SLURM_JOB_NAME:-LOCAL_RUN}

# ==============================
# CONFIG
# ==============================
DATA_DIR="/dev/shm/DATA/BridgeTrain"   # Ensure your zip files are extracted here
ENV_DIR="$PROJECT_DIR/myenv"

DEVKIT_DIR="$PROJECT_DIR/datasets/DOTA_devkit"
CKPT_DIR="$PROJECT_DIR/weights_dota"

TARGET_EPOCH=50
EPOCHS_PER_SESSION=2
PHASE1_EPOCHS=10

TRAINVAL_FILE="$CKPT_DIR/trainval.txt"

# ==============================
# LOGGING
# ==============================
echo "========================================"
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "========================================"

start_time=$(date +%s)

# ==============================
# ENV SETUP
# ==============================
if [[ -f "$ENV_DIR/bin/activate" ]]; then
  source "$ENV_DIR/bin/activate"
else
  echo "WARNING: venv not found at $ENV_DIR. Script might fail if dependencies are missing."
fi

export PYTHONPATH="$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}"

echo "Using Python: $(which python)"
# (Removed the PyTorch test here to prevent OOM Killer crashes before training)

# ==============================
# VERIFY PATHS
# ==============================
[[ -f "$PROJECT_DIR/main.py" ]] || { echo "ERROR: main.py not found"; exit 1; }
[[ -d "$DEVKIT_DIR" ]] || { echo "ERROR: devkit missing"; exit 1; }
[[ -d "$CKPT_DIR" ]] || mkdir -p "$CKPT_DIR"

# ==============================
# BUILD EXTENSION
# ==============================
if ! ls "$DEVKIT_DIR"/polyiou*.so &>/dev/null; then
  echo "Building polyiou..."
  cd "$DEVKIT_DIR"
  swig -c++ -python polyiou.i
  python setup.py build_ext --inplace
  cd "$PROJECT_DIR"
fi

# ==============================
# HELPERS
# ==============================
get_latest_epoch() {
  local latest=0
  shopt -s nullglob
  for f in "$CKPT_DIR"/model_*.pth; do
    n=${f##*/model_}
    n=${n%.pth}
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > latest )); then
      latest=$n
    fi
  done
  shopt -u nullglob
  echo "$latest"
}

archive_trainval() {
  local start="$1"
  local end="$2"

  [[ ! -f "$TRAINVAL_FILE" ]] && return

  local dst="$CKPT_DIR/trainval_${start}to${end}.txt"
  mv "$TRAINVAL_FILE" "$dst"
  echo "Archived: $dst"
}

resubmit() {
  local epoch="$1"

  if (( epoch >= TARGET_EPOCH )); then
    echo "Target reached."
    return
  fi

  if [[ $IS_SLURM -eq 1 ]] && command -v sbatch &>/dev/null; then
    echo "Resubmitting to SLURM queue..."
    sbatch "$0"
  else
    echo "Not running in SLURM queue. Stopping session here."
    echo "To continue training, run this script again."
  fi
}

# ==============================
# TRAIN LOGIC
# ==============================
current_epoch=$(get_latest_epoch)
echo "Current epoch: $current_epoch"

if (( current_epoch >= TARGET_EPOCH )); then
  echo "Training already complete."
  exit 0
fi

session_end=$(( current_epoch + EPOCHS_PER_SESSION ))

if (( current_epoch < PHASE1_EPOCHS && session_end > PHASE1_EPOCHS )); then
  session_end=$PHASE1_EPOCHS
fi

(( session_end > TARGET_EPOCH )) && session_end=$TARGET_EPOCH

echo "Training to epoch: $session_end"

# ==============================
# BUILD COMMAND
# ==============================
CMD=(
  python main.py
  --data_dir "$DATA_DIR"
  --num_epoch "$session_end"
  --batch_size 10
  --dataset dota
  --phase train
  --conf_thresh 0.1
)

if (( current_epoch == 0 )); then
  CMD+=(--pretrained --heatmap_only)

elif (( current_epoch < PHASE1_EPOCHS )); then
  CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth" --heatmap_only)

elif (( current_epoch == PHASE1_EPOCHS )); then
  CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth" --reset_lr)

else
  CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth")
fi

# ==============================
# RUN
# ==============================
echo "Running training..."
"${CMD[@]}"

# ==============================
# POST
# ==============================
new_epoch=$(get_latest_epoch)
echo "New epoch: $new_epoch"

if (( new_epoch > current_epoch )); then
  archive_trainval "$((current_epoch+1))" "$new_epoch"
fi

resubmit "$new_epoch"

end_time=$(date +%s)

echo "========================================"
echo "End: $(date)"
echo "Runtime: $((end_time - start_time)) sec"
echo "========================================"