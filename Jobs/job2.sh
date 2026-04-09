#!/bin/bash
#SBATCH --job-name=glh_imp
#SBATCH --output=efv2_train_test2.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $(hostname)"
echo "Submit time: $(scontrol show job $SLURM_JOB_ID | grep SubmitTime | awk -F= '{print $2}')"
echo "Start time: $(date)"
echo "========================================"

start_time=$(date +%s)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /media02/hvtham/conda_envs/myenv

PROJECT_DIR=/media02/hvtham/BBAV/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit
CKPT_DIR=$PROJECT_DIR/weights_dota
TARGET_EPOCH=50
EPOCHS_PER_SESSION=2
PHASE1_EPOCHS=10       # epochs trained with heatmap-only loss
TRAINVAL_FILE=$CKPT_DIR/trainval.txt

get_latest_epoch() {
  local latest=0
  local f base n

  shopt -s nullglob
  for f in "$CKPT_DIR"/model_*.pth; do
    base=$(basename "$f")
    n=${base#model_}
    n=${n%.pth}
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > latest )); then
      latest=$n
    fi
  done
  shopt -u nullglob

  echo "$latest"
}

RESUBMITTED=0
resubmit_if_needed() {
  local latest_epoch="$1"

  if (( RESUBMITTED == 1 )); then
    return
  fi

  if (( latest_epoch < TARGET_EPOCH )); then
    echo "Latest epoch is $latest_epoch (target: $TARGET_EPOCH). Resubmitting..."
    sbatch "$0"
    RESUBMITTED=1
  else
    echo "Latest epoch is $latest_epoch. Target reached; no resubmit needed."
  fi
}

archive_trainval_if_present() {
  local start_epoch="$1"
  local end_epoch="$2"
  local dst
  local suffix=1

  if [[ ! -f "$TRAINVAL_FILE" ]]; then
    return
  fi

  dst="$CKPT_DIR/trainval_${start_epoch}to${end_epoch}.txt"
  while [[ -e "$dst" ]]; do
    dst="$CKPT_DIR/trainval_${start_epoch}to${end_epoch}_$suffix.txt"
    ((suffix++))
  done

  mv "$TRAINVAL_FILE" "$dst"
  echo "Archived trainval log: $dst"
}

handle_pre_timeout() {
  echo "Received pre-time-limit signal. Preparing immediate resubmission..."
  local latest_epoch
  latest_epoch=$(get_latest_epoch)
  resubmit_if_needed "$latest_epoch"
  exit 0
}

trap handle_pre_timeout USR1 TERM

cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA available check:"
python -c "import torch; print(torch.cuda.is_available())"

if [ ! -f "$DEVKIT_DIR/polyiou.cpython-*.so" ]; then
    echo "Building polyiou..."
    cd $DEVKIT_DIR
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
    cd $PROJECT_DIR
fi

current_epoch=$(get_latest_epoch)
echo "Latest checkpoint epoch before run: $current_epoch"

if (( current_epoch >= TARGET_EPOCH )); then
  echo "Target epoch $TARGET_EPOCH already reached. Exiting without training."
  exit 0
fi

# Cap session end at phase boundary if still in phase 1, to avoid crossing phases mid-session
session_end_epoch=$(( current_epoch + EPOCHS_PER_SESSION ))
if (( current_epoch < PHASE1_EPOCHS && session_end_epoch > PHASE1_EPOCHS )); then
  session_end_epoch=$PHASE1_EPOCHS
fi
if (( session_end_epoch > TARGET_EPOCH )); then
  session_end_epoch=$TARGET_EPOCH
fi

TRAIN_CMD=(
  main.py
  --data_dir /media02/hvtham/DATA/BridgeTrain
  --num_epoch "$session_end_epoch"
  --batch_size 5
  --dataset dota
  --phase train
  --conf_thresh 0.1
)

if (( current_epoch == 0 )); then
  echo "Phase 1 — epoch 1 to $session_end_epoch — heatmap-only loss, pretrained backbone"
  TRAIN_CMD+=(--pretrained --heatmap_only)
elif (( current_epoch < PHASE1_EPOCHS )); then
  echo "Phase 1 — resuming epoch $((current_epoch+1)) to $session_end_epoch — heatmap-only loss"
  TRAIN_CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth" --heatmap_only)
elif (( current_epoch == PHASE1_EPOCHS )); then
  echo "Phase 2 start — resuming from epoch $current_epoch — full loss, LR reset to fresh schedule"
  TRAIN_CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth" --reset_lr)
else
  echo "Phase 2 — resuming epoch $((current_epoch+1)) to $session_end_epoch — full loss"
  TRAIN_CMD+=(--resume_train "$CKPT_DIR/model_${current_epoch}.pth")
fi

echo "This session will train to epoch $session_end_epoch (target: $TARGET_EPOCH)"

python "${TRAIN_CMD[@]}"

latest_epoch=$(get_latest_epoch)
echo "Latest checkpoint epoch after run: $latest_epoch"

if (( latest_epoch > current_epoch )); then
  archive_trainval_if_present "$((current_epoch + 1))" "$latest_epoch"
fi

resubmit_if_needed "$latest_epoch"

end_time=$(date +%s)

echo "========================================"
echo "End time: $(date)"
echo "Total runtime: $((end_time - start_time)) seconds"
echo "========================================"
