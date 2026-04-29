#!/bin/bash
# Evaluation script for a single rented machine (no SLURM).
# Run inside tmux: tmux new-session -s eval 'bash Jobs/job13.sh'

set -euo pipefail

echo "========================================"
echo "PID: $$"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "========================================"

start_time=$(date +%s)

PROJECT_DIR=/workspace/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit
WEIGHTS_DIR=${BBAV_WEIGHTS_DIR:-${BBAV_SAVE_DIR:-/dev/shm}/weights_dota}
OUTPUT_DIR=$PROJECT_DIR/eval_results

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

for i in $(seq 11 50); do
  echo "========================================"
  echo "Evaluating model_${i}.pth"
  echo "========================================"

  WEIGHTS="$WEIGHTS_DIR/model_${i}.pth"
  if [ ! -f "$WEIGHTS" ]; then
    echo "Missing weights: $WEIGHTS (skipping)"
    continue
  fi

  python main.py \
    --data_dir "/workspace/DATA/Validate_DOTA_1_0.5" \
    --batch_size 15 \
    --dataset dota \
    --phase eval \
    --conf_thresh 0.1 \
    --resume "$WEIGHTS"

  echo "Running DOTA evaluation..."

  cd "$DEVKIT_DIR"
  python dota_evaluation_task1.py > "$OUTPUT_DIR/eval_model_${i}.txt"
  cd "$PROJECT_DIR"

  echo "Saved: $OUTPUT_DIR/eval_model_${i}.txt"
done

end_time=$(date +%s)

echo "========================================"
echo "Finished all evaluations"
echo "End: $(date)"
echo "Total runtime: $((end_time - start_time)) seconds"
echo "Results saved in $OUTPUT_DIR"
echo "========================================"
