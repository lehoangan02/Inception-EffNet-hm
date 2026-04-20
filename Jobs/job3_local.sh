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
WEIGHTS_DIR="${WEIGHTS_DIR:-$PROJECT_DIR/weights_dota/}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_results}"
DATA_DIR="${DATA_DIR:-/Volumes/ExternalSSD/data/glhtestsplit}"
COREML_MODEL=${COREML_MODEL:-}
COREML_COMPUTE_UNITS=${COREML_COMPUTE_UNITS:-cpu_and_ne}
MERGE_DIR="${MERGE_DIR:-$PROJECT_DIR/merge_dota}"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

python -c "import torch; print('CUDA:', torch.cuda.is_available())"



for i in $(seq 9 10)
do

if [[ "$(uname -s)" == "Darwin" ]]; then
  COREML_MODEL="${COREML_MODEL:-$WEIGHTS_DIR/model_$i.mlpackage}"
  if [[ ! -f "$COREML_MODEL" ]]; then
    COREML_CHECKPOINT="${COREML_CHECKPOINT:-$WEIGHTS_DIR/model_$i.pth}"
    if [[ ! -f "$COREML_CHECKPOINT" ]]; then
      echo "Core ML checkpoint not found at $COREML_CHECKPOINT"
      exit 1
    fi
    echo "Core ML model not found at $COREML_MODEL"
    echo "Exporting Core ML model from $COREML_CHECKPOINT..."
    python export_coreml.py --checkpoint "$COREML_CHECKPOINT" --output "$COREML_MODEL"
  fi
  BACKEND_ARGS=(--backend coreml --coreml_model "$COREML_MODEL" --coreml_compute_units "$COREML_COMPUTE_UNITS")
  echo "Using Core ML backend with model: $COREML_MODEL"
else
  BACKEND_ARGS=(--backend pytorch)
  echo "Using PyTorch backend"
fi

echo "========================================"
echo "Evaluating model_$i.pth"
echo "========================================"

WEIGHTS=$WEIGHTS_DIR/model_${i}.pth

if [[ ! -f "$WEIGHTS" && -z "$COREML_MODEL" ]]; then
  echo "Skipping model_$i.pth (not found at $WEIGHTS)"
  continue
fi

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
  rm -f "Task1_bridge.txt"
else
  echo "Warning: $MERGE_DIR/Task1_bridge.txt not found after model_$i inference"
fi

cd "$PROJECT_DIR"

done

echo "========================================"
echo "Finished all evaluations"
echo "End: $(date)"
echo "Results saved in $OUTPUT_DIR"
echo "========================================"