# Improving Oriented Object Detection in Aerial Images with Inception-Enhanced EfficientNetV2

As remote sensing and aerial imagery technologies rapidly evolve, the demand for highly accurate and efficient oriented object detection remains a prominent challenge. While current baseline models evaluated on the DOTA dataset provide a solid foundation, they often lack the receptive diversity necessary to resolve densely packed objects and extreme scale variations inherent in aerial views. In this paper, we propose a structurally adapted anchor-free framework that systematically addresses these spatial limitations. Specifically, we leverage compound scaling via EfficientNetV2 and integrate a multi-scale spatial feature aggregation module into the Box Parameters head to capture diverse receptive fields. Crucially, we establish a rigorous, task-specific heatmap pretraining strategy focused strictly on object center localization. This calibrated training protocol serves as a strong methodological foundation, providing a robust initialization that significantly minimizes early-stage misclassification errors. Comprehensive experiments demonstrate that these problem-driven modifications collectively improve the mean Average Precision (mAP) on the DOTA dataset by 4.75\% over the baseline. Our findings underscore the distinct advantages of pairing calibrated training protocols with targeted architectural adaptations to achieve robust aerial object detection.

<p align="center">
	<img src="imgs/diagram.png", width="800">
</p>

# Validation Results on [DOTA-v1.0](https://captain-whu.github.io/DOTA/index.html)

The model weights can be downloaded from the following links: [Baseline](https://huggingface.co/datasets/rabbitKabbit/attempt2_batchsize20/resolve/main/model_50.pth), [Ours](https://huggingface.co/datasets/rabbitKabbit/attempt7_hm5/resolve/main/model_48.pth)

```ruby
## Baseline: model_50.pth
mAP: 0.6458690174873016
ap of each class: plane:89.3007906, baseball-diamond:70.66393795, bridge:36.75460756, ground-track-field:61.09327851, small-vehicle:30.44215374, large-vehicle:71.80267091, ship:85.50325898, tennis-court:90.84194978, basketball-court:66.59185562, storage-tank:78.86206304, soccer-ball-field:63.40755401, roundabout:58.71973467, harbor:63.22163028, swimming-pool:57.33294037, helicopter:44.26510022

## Ours: model_48.pth
mAP: 0.7373316196409346
ap of each class: plane:89.69814021, baseball-diamond:75.26651978, bridge:51.1799739, ground-track-field:66.04550627, small-vehicle:67.73603774, large-vehicle:83.73694647, ship:87.70576262, tennis-court:90.88199431, basketball-court:72.02409421, storage-tank:88.41285061, soccer-ball-field:73.4644523, roundabout:72.11833638, harbor:68.59598763, swimming-pool:64.55090114, helicopter:54.57992587
```


# Dependencies
Ubuntu 18.04, Python 3.6.10, PyTorch 1.6.0, OpenCV-Python 4.3.0.36 

# How To Start

Download and follow the installation instructions for the DOTA development kit [DOTA_devkit](https://anonymous.4open.science/r/DOTA_devkit-0FE1) and put it under datasets folder.

## GPU Notes
- Use a CUDA-enabled PyTorch build so `torch.cuda.is_available()` returns true.
- Weights are stored under `weights_<dataset>` inside `--save_dir` (default: `BBAV_SAVE_DIR`, then `/dev/shm` if writable, else current directory).
- Use `--pretrained` to download ImageNet weights for the backbone; use `--no-pretrained` on offline machines.
- Optional flags for debugging: `--heatmap_only` and `--phase loss` (see `--loss_epochs`).

## About DOTA

### Split Image
Split the DOTA images from [DOTA_devkit](https://anonymous.4open.science/r/DOTA_devkit-0FE1) before training, testing and evaluation.

The dota ```trainval``` and ```test``` datasets are cropped into ```608x608``` patches with a stride of `100` and two scales `0.5` and `1`.

The splitted DOTA validation dataset can be found [here](https://huggingface.co/datasets/rabbitKabbit/BridgeTrain/resolve/main/Cross%20Validation/Validate_DOTA_1_0.5.zip?download=true)

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

## Data Arrangement

### DOTA
```
data_dir/
        images/*.png
        labelTxt/*.txt
        trainval.txt
        test.txt
```
you may modify `datasets/dataset_dota.py` to adapt code to your own data.

## Quick GPU Sanity Check (Small Batch)
Create a tiny dataset subset (e.g. 5-20 images/labels) and update `trainval.txt`/`test.txt` to point to it. Then run:
```ruby
python main.py --data_dir dataPath --epochs 1 --batch_size 1 --dataset dota --phase train --save_dir ./runs --pretrained
# python main.py --data_dir dataPath --batch_size 1 --dataset dota --phase test --resume model_1.pth
python main.py --data_dir dataPath --conf_thresh 0.1 --batch_size 1 --dataset dota --phase eval --resume model_1.pth
```

## Train Model
```ruby
python main.py --data_dir dataPath --epochs 50 --batch_size 16 --dataset dota --phase train
```

<!-- ## Test Model
```ruby
python main.py --data_dir dataPath --batch_size 16 --dataset dota --phase test
``` -->

## Evaluate Model
```ruby
python main.py --data_dir dataPath --conf_thresh 0.1 --batch_size 16 --dataset dota --phase eval
```

You may change `conf_thresh` to get a better `mAP`. 

Please zip and upload the generated `merge_dota` for DOTA [Task1](https://captain-whu.github.io/DOTA/evaluation.html) evaluation.
