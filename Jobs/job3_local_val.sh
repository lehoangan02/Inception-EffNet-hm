#!/bin/bash

set -euo pipefail

echo "========================================"
echo "Running local evaluation"
echo "Machine: $(hostname)"
echo "Start: $(date)"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEVKIT_DIR="${DEVKIT_DIR:-$PROJECT_DIR/datasets/DOTA_devkit}"
WEIGHTS_DIR="/Volumes/ExternalSSD/data/weights_dota/attempt_pretrain5"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_results}"
MAP_DIR_NAME="${MAP_DIR_NAME:-608_attempt_pretrain_5}"
MAP_OUTPUT_DIR="${MAP_OUTPUT_DIR:-$OUTPUT_DIR/$MAP_DIR_NAME}"
DATA_DIR="${DATA_DIR:-/Volumes/ExternalSSD/data/Validate_DOTA_1_0.5}"
LABEL_DIR="${LABEL_DIR:-/Volumes/ExternalSSD/data/labelTxt}"
IMAGESET_FILE="${IMAGESET_FILE:-/Volumes/ExternalSSD/data/val_orig.txt}"
COREML_MODEL=${COREML_MODEL:-}
COREML_COMPUTE_UNITS=${COREML_COMPUTE_UNITS:-cpu_and_ne}
MERGE_DIR="${MERGE_DIR:-$PROJECT_DIR/merge_dota}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MAP_OUTPUT_DIR"

SUMMARY_FILE="$MAP_OUTPUT_DIR/map_summary.txt"
echo "Run started: $(date)" > "$SUMMARY_FILE"
echo "MAP output directory: $MAP_OUTPUT_DIR" >> "$SUMMARY_FILE"

cd "$PROJECT_DIR"

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}
export MERGE_DIR LABEL_DIR IMAGESET_FILE

python -c "import torch; print('CUDA:', torch.cuda.is_available())"



for i in $(seq 5 9)
do

WEIGHTS=$WEIGHTS_DIR/model_${i}.pth
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Skipping model_$i.pth (not found at $WEIGHTS)"
  continue
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  CURRENT_COREML_MODEL="$WEIGHTS_DIR/model_$i.mlpackage"
  if [[ ! -f "$CURRENT_COREML_MODEL" ]]; then
    echo "Core ML model not found at $CURRENT_COREML_MODEL"
    echo "Exporting Core ML model from $WEIGHTS..."
    python export_coreml.py --checkpoint "$WEIGHTS" --output "$CURRENT_COREML_MODEL"
  fi

  if [[ -n "$COREML_MODEL" ]]; then
    CURRENT_COREML_MODEL="$COREML_MODEL"
    if [[ ! -f "$CURRENT_COREML_MODEL" ]]; then
      echo "COREML_MODEL override not found at $CURRENT_COREML_MODEL"
      exit 1
    fi
  fi

  BACKEND_ARGS=(--backend coreml --coreml_model "$CURRENT_COREML_MODEL" --coreml_compute_units "$COREML_COMPUTE_UNITS")
  echo "Using Core ML backend with model: $CURRENT_COREML_MODEL"
else
  BACKEND_ARGS=(--backend pytorch)
  echo "Using PyTorch backend"
fi

echo "========================================"
echo "Evaluating model_$i.pth"
echo "========================================"

python main.py \
  --data_dir "$DATA_DIR" \
  --batch_size 16 \
  --dataset dota \
  --phase eval \
  --conf_thresh 0.1 \
  --resume "$WEIGHTS" \
  "${BACKEND_ARGS[@]}"

if [[ -f "$MERGE_DIR/Task1_bridge.txt" ]]; then
  echo "Running bridge postprocess for model_$i..."
  cd "$MERGE_DIR"
  python glh_postprocess.py Task1_bridge.txt Task1_bridge.txt
  zip -j "Task1_bridge_${i}.zip" "Task1_bridge.txt"
else
  echo "Warning: $MERGE_DIR/Task1_bridge.txt not found after model_$i inference"
fi

echo "Running DOTA mAP evaluation for model_$i..."
cd "$DEVKIT_DIR"
MODEL_EVAL_FILE="$MAP_OUTPUT_DIR/eval_model_${i}.txt"
python dota_evaluation_task1.py > "$MODEL_EVAL_FILE"
MAP_LINE="$(grep -E '^map:' "$MODEL_EVAL_FILE" | tail -n 1 || true)"
if [[ -n "$MAP_LINE" ]]; then
  echo "model_${i} ${MAP_LINE}" | tee -a "$SUMMARY_FILE"
fi

if [[ -f "$MERGE_DIR/Task1_bridge.txt" ]]; then
  rm -f "$MERGE_DIR/Task1_bridge.txt"
fi

cd "$PROJECT_DIR"

done

echo "========================================"
echo "Finished all evaluations"
echo "End: $(date)"
echo "Results saved in $OUTPUT_DIR"
echo "mAP files saved in $MAP_OUTPUT_DIR"
echo "========================================"
