#!/bin/bash
cd ~/LHAN/Inception-EffNet-hm
. ../hrsc_venv/bin/activate
export PYTHONPATH=$PWD/datasets/DOTA_devkit
while pgrep -f "gdown 1_1ov" >/dev/null; do sleep 5; done
W=$(ls weights_hrsc/*.pth | head -1); echo weights $W
for P in test val; do
  echo "=== $P start $(date +%T)"
  EVAL_PHASE=$P python main.py --data_dir ../hrsc_data/HRSC2016 --conf_thresh 0.1 --batch_size 1 --dataset hrsc --phase eval --resume $(basename $W) --save_dir . > eval_hrsc_$P.log 2>&1
  echo "=== $P end $(date +%T)"; grep -E "^map|ship:" eval_hrsc_$P.log
done
