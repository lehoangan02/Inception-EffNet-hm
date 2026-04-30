# Improving Oriented Object Detection in Aerial Images with Inception-Enhanced EfficientNetV2

As remote sensing and aerial imagery technologies rapidly evolve, the demand for highly accurate and efficient oriented object detection remains a prominent challenge. While current baseline models evaluated on the DOTA dataset provide a solid foundation, their performance often requires structural refinement for practical, high-precision applications. In this paper, we propose an optimized adaptation of the anchor-free BBAVectors framework. Specifically, we integrate the EfficientNetV2 backbone to enhance feature extraction and embed an Inception module into the Box Parameters prediction head to enable multi-scale spatial feature aggregation. Additionally, we introduce a task-specific heatmap pretraining strategy focused solely on object center localization, which effectively minimizes early-stage misclassification errors. Comprehensive experiments demonstrate that these targeted modifications collectively improve the mean Average Precision (mAP) on the DOTA dataset by 6.98% over the baseline method. Our findings underscore the distinct advantages of pairing advanced backbone architectures with specialized prediction heads and calibrated training protocols to achieve robust oriented object detection.

<p align="center">
	<img src="imgs/diagram.png", width="800">
</p>

# Evaluation Results on [DOTA-v1.0](https://captain-whu.github.io/DOTA/evaluation.html)

The model weights can be downloaded from the following links: [Baseline](https://huggingface.co/datasets/lehoangan02/attempt1_batchsize15/resolve/main/model_48.pth), [Ours](https://huggingface.co/datasets/lehoangan02/attempt6_hm10/resolve/main/model_50.pth)

```ruby
## Baseline: model_48.pth
mAP: 0.7536283690546086
ap of each class: plane:0.8862514770737425, baseball-diamond:0.8406009896282075, bridge:0.521285610860641, ground-track-field:0.6955552280263699, small-vehicle:0.7825702607967113, large-vehicle:0.8040010247209182, ship:0.8805575982076236, tennis-court:0.9087489402165854, basketball-court:0.8722663525600673, storage-tank:0.8638699841268725, soccer-ball-field:0.5610545208583243, roundabout:0.6562139014619145, harbor:0.6709747110284013, swimming-pool:0.7208480121858474, helicopter:0.6396269240669054

## Ours: model_50.pth
mAP: 0.8234793611602195
ap of each class: plane:0.8859121197958046, baseball-diamond:0.8483251642688572, bridge:0.5214374843409882, ground-track-field:0.6560710395759289, small-vehicle:0.7773671634218439, large-vehicle:0.7427879633964128, ship:0.8804625721887132, tennis-court:0.908816372618596, basketball-court:0.862399364058993, storage-tank:0.8670730838290734, soccer-ball-field:0.5987801663737911, roundabout:0.6401450110418495, harbor:0.6698206063852568, swimming-pool:0.7071826121359568, helicopter:0.672510279226682
```


# Dependencies
Ubuntu 18.04, Python 3.6.10, PyTorch 1.6.0, OpenCV-Python 4.3.0.36 

# How To Start

Download and install the DOTA development kit [DOTA_devkit](https://github.com/lehoangan02/DOTA_devkit) and put it under datasets folder.

## GPU Notes
- Use a CUDA-enabled PyTorch build so `torch.cuda.is_available()` returns true.
- Weights are stored under `weights_<dataset>` inside `--save_dir` (default: `BBAV_SAVE_DIR`, then `/dev/shm` if writable, else current directory).
- Use `--pretrained` to download ImageNet weights for the backbone; use `--no-pretrained` on offline machines.
- Optional flags for debugging: `--heatmap_only` and `--phase loss` (see `--loss_epochs`).

## About DOTA
### Split Image
Split the DOTA images from [DOTA_devkit](https://github.com/lehoangan02/DOTA_devkit) before training, testing and evaluation.

The dota ```trainval``` and ```test``` datasets are cropped into ```608x608``` patches with a stride of `100` and two scales `0.5` and `1`.

### About Split TXT Files
The `trainval.txt` and `test.txt` used in `datasets/dataset_dota.py` contain the list of image names without suffix, example:
```
P0000__0.5__0___0
P0000__0.5__0___1000
P0000__0.5__0___1500
P0000__0.5__0___2000
P0000__0.5__0___2151
P0000__0.5__0___500
P0000__0.5__1000___0
```
Format of the ground-truth for DOTA dataset: `x1, y1, x2, y2, x3, y3, x4, y4, category, difficulty`

Examples: 
```
275.0 463.0 411.0 587.0 312.0 600.0 222.0 532.0 tennis-court 0
341.0 376.0 487.0 487.0 434.0 556.0 287.0 444.0 tennis-court 0
428.0 6.0 519.0 66.0 492.0 108.0 405.0 50.0 bridge 0
```
## Data Arrangment
### DOTA
```
data_dir/
        images/*.png
        labelTxt/*.txt
        trainval.txt
        test.txt
```
you may modify `datasets/dataset_dota.py` to adapt code to your own data.
## Train Model
```ruby
python main.py --data_dir dataPath --epochs 80 --batch_size 16 --dataset dota --phase train
```

## Quick GPU Sanity Check (Small Batch)
Create a tiny dataset subset (e.g. 5-20 images/labels) and update `trainval.txt`/`test.txt` to point to it. Then run:
```ruby
python main.py --data_dir dataPath --epochs 1 --batch_size 2 --num_workers 0 --dataset dota --phase train --save_dir ./runs --pretrained
python main.py --data_dir dataPath --batch_size 1 --dataset dota --phase test --save_dir ./runs --resume model_1.pth
```

## Test Model
```ruby
python main.py --data_dir dataPath --batch_size 16 --dataset dota --phase test
```


<!-- ## Evaluate Model
```ruby
python main.py --data_dir dataPath --conf_thresh 0.1 --batch_size 16 --dataset dota --phase eval
``` -->

You may change `conf_thresh` to get a better `mAP`. 

Please zip and upload the generated `merge_dota` for DOTA [Task1](https://captain-whu.github.io/DOTA/evaluation.html) evaluation.
