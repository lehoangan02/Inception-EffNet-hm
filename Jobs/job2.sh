```bash
#!/bin/bash

# ==============================
# SAFE SETTINGS
# ==============================
set -eo pipefail  # removed -u to avoid crashing on missing SLURM vars

# ==============================
# DETECT ENVIRONMENT
# ==============================
IS_SLURM=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  IS_SLURM=1
fi

JOB_ID=${SLURM_JOB_ID:-LOCAL}
JOB_NAME=${SLURM_JOB_NAME:-LOCAL_RUN}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}

# ==============================
# PATH CONFIG
# ==============================
PROJECT_DIR="."
DATA_DIR="../DATA/BridgeTrain"
ENV_DIR="./myenv"

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
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Start time: $(date)"

if [[ $IS_SLURM -eq 1 ]]; then
  echo "Submit time: $(scontrol show job $SLURM_JOB_ID | grep SubmitTime | awk -F= '{print $2}')"
else
  echo "Submit time: N/A (local run)"
fi
echo "========================================"

start_time=$(date +%s)

# ==============================
# ACTIVATE ENV
# ==============================
if [[ -f "$ENV_DIR/bin/activate" ]]; then
  source "$ENV_DIR/bin/activate"
else
  echo "WARNING: Virtualenv not found at $ENV_DIR"
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

archive_trainval_if_present() {
  local start_epoch="$1"
  local end_epoch="$2"

  [[ ! -f "$TRAINVAL_FILE" ]] && return

  local dst="$CKPT_DIR/trainval_${start_epoch}to${end_epoch}.txt"
  local suffix=1

  while [[ -e "$dst" ]]; do
    dst="$CKPT_DIR/trainval_${start_epoch}to${end_epoch}_$suffix.txt"
    ((suffix++))
  done

  mv "$TRAINVAL_FILE" "$dst"
  echo "Archived: $dst"
}

resubmit_if_needed() {
  local latest_epoch="$1"

  if (( latest_epoch >= TARGET_EPOCH )); then
    echo "Target reached ($latest_epoch). No resubmit."
    return
  fi

  if [[ $IS_SLURM -eq 1 ]] && command -v sbatch &> /dev/null; then
    echo "Resubmitting via SLURM..."
    sbatch "$0"
  else
    echo "Skipping resubmit (no SLURM)"
  fi
}

handle_pre_timeout() {
  echo "Received timeout signal"
  latest_epoch=$(get_latest_epoch)
  resubmit_if_needed "$latest_epoch"
  exit 0
}

trap handle_pre_timeout USR1 TERM

# ==============================
# SETUP
# ==============================
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}"

echo "Python: $(which python)"
echo "CUDA check:"
python -c "import torch; print(torch.cuda.is_available())"

# ==============================
# BUILD EXTENSION (if needed)
# ==============================
if ! ls "$DEVKIT_DIR"/polyiou*.so &>/dev/null; then
  echo "Building polyiou..."
  cd "$DEVKIT_DIR"
  swig -c++ -python polyiou.i
  python setup.py build_ext --inplace
  cd - > /dev/null
fi

# ==============================
# TRAINING LOGIC
# ==============================
current_epoch=$(get_latest_epoch)
echo "Current epoch: $current_epoch"

if (( current_epoch >= TARGET_EPOCH )); then
  echo "Already reached target. Exiting."
  exit 0
fi

session_end=$(( current_epoch + EPOCHS_PER_SESSION ))

# Prevent crossing phase boundary
if (( current_epoch < PHASE1_EPOCHS && session_end > PHASE1_EPOCHS )); then
  session_end=$PHASE1_EPOCHS
fi

(( session_end > TARGET_EPOCH )) && session_end=$TARGET_EPOCH

echo "Training to epoch: $session_end"

# ==============================
# BUILD TRAIN COMMAND
# ==============================
CMD=(
  python main.py
  --data_dir "$DATA_DIR"
  --num_epoch "$session_end"
  --batch_size 5
  --dataset dota
  --phase train
  --conf_thresh 0.1
)

if (( current_epoch == 0 )); then
  echo "Phase 1 start"
  CMD+=(--pretrained --heatmap_only)

elif (( current_epoch < PHASE1_EPOCHS )); then
  echo "Phase 1 resume"
  CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth" --heatmap_only)

elif (( current_epoch == PHASE1_EPOCHS )); then
  echo "Phase 2 start"
  CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth" --reset_lr)

else
  echo "Phase 2 resume"
  CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth")
fi

# ==============================
# RUN TRAINING
# ==============================
"${CMD[@]}"

latest_epoch=$(get_latest_epoch)
echo "New epoch: $latest_epoch"

if (( latest_epoch > current_epoch )); then
  archive_trainval_if_present "$((current_epoch+1))" "$latest_epoch"
fi

resubmit_if_needed "$latest_epoch"

# ==============================
# END
# ==============================
end_time=$(date +%s)

echo "========================================"
echo "End time: $(date)"
echo "Runtime: $((end_time - start_time)) sec"
echo "========================================"
```
