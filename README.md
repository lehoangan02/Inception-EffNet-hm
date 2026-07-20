# Multi-Scale Feature Aggregation and Center-Localization Pretraining for Oriented Object Detection

As remote sensing and aerial imagery technologies rapidly evolve, the demand for highly accurate oriented object detection remains a prominent challenge. While current baseline models evaluated on the DOTA dataset provide a solid foundation, they often lack the receptive diversity necessary to resolve densely packed objects and extreme scale variations inherent in aerial views. In this paper, we propose a structurally adapted anchor-free framework that systematically addresses these spatial limitations. We introduce a multi-scale feature aggregation module (MSAM) within the Box Parameters head to resolve extreme aspect ratios, coupled with a calibrated initialization protocol that overcomes the optimization instability typical of anchor-free models. Crucially, we establish a task-specific heatmap pretraining strategy focused strictly on object center localization to minimize early-stage misclassification errors. Comprehensive experiments on the DOTA dataset demonstrate the data efficiency of our approach. Trained exclusively on the standard training split, our model achieves 75.98\% mAP, outperforming the peak performance of the standard Box Boundary-Aware Vectors baseline at 75.36\% mAP which requires training on the expanded train-and-validation splits. Our findings underscore the advantages of pairing calibrated training protocols with targeted architectural adaptations for data-efficient oriented object detection.

<p align="center">
	<img src="imgs/diagram.png", width="800">
</p>

# Testing Results on [DOTA-v1.0](https://captain-whu.github.io/DOTA/index.html)

The model weights can be downloaded from the following links: [Baseline](https://drive.google.com/file/d/1uqb1hTcdzsx3xZnIWoGXSmEIThWkOADp/view?usp=drive_link), [Ours](https://drive.google.com/file/d/1gl48egGwBE2JUJ_Ll_zpUyN4fqfGYniL/view?usp=sharing)

```ruby
## Baseline: model_50.pth
mAP: 0.7536283690546086
ap of each class: plane:88.62514771, baseball-diamond:84.06009896, bridge:52.12856109, ground-track-field:69.55552280, small-vehicle:78.25702608, large-vehicle:80.40010247, ship:88.05575982, tennis-court:90.87489402, basketball-court:87.22663526, storage-tank:86.38699841, soccer-ball-field:56.10545209, roundabout:65.62139015, harbor:67.09747110, swimming-pool:72.08480122, helicopter:63.96269241

## Ours: model_73.pth
mAP: 0.7597520192929833
ap of each class: plane:88.65230014, baseball-diamond:84.77466462, bridge:54.70068208, ground-track-field:69.77795239, small-vehicle:79.34679051, large-vehicle:83.50117096, ship:87.45518086, tennis-court:90.88184315, basketball-court:86.82064228, storage-tank:86.60548695, soccer-ball-field:55.16758990, roundabout:73.49078311, harbor:65.86082071, swimming-pool:72.55930986, helicopter:60.03281142
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
