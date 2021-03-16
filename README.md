# DFC2021 MSD Baseline
This repo contains implementations of several baseline for the ["Multitemporal Semantic Change Detection" (MSD) track](http://www.grss-ieee.org/community/technical-committees/data-fusion/2021-ieee-grss-data-fusion-contest-track-msd/) of the 2021 IEEE GRSS Data Fusion Competition (DFC2021). See the [CodaLab page](https://competitions.codalab.org/competitions/27956) for more information about the competition, including the current leaderboard!


## Project description
.
├── add_single_cls.py # 
├── boost_inference.py
├── create_nlcd_only_baseline.py
├── data
│   ├── dfc2021_index.geojson
│   ├── dfc2021_index.txt
│   ├── splits
│   │   ├── test_inference.csv
│   │   ├── test_origin_label_radio_2.csv
│   │   ├── training_set_naip_nlcd_2013.csv
│   │   ├── training_set_naip_nlcd_2017.csv
│   │   ├── training_set_naip_nlcd_both.csv
│   │   └── val_inference_both.csv
│   ├── test_tiles.txt
│   └── val_tiles.txt
├── dataloaders
│   ├── Landsat2NlcdDatasets.py
│   ├── Landsat2NlcdTileDatasets.py
│   ├── StreamingDatasets.py
│   ├── TileDatasets.py
│   ├── TileMLDatasets.py
│   └── __init__.py
├── independent_pairs_to_predictions.py
├── inference.py
├── label_inference.py
├── landsat2nlcd_inference.py
├── models.py
├── predictions_clean.py
├── rfc.py
├── single_class_data_maker.py
├── single_inference.py
├── single_train.py
├── train.py
├── train_landsat2nlcd.py
├── tree.text
├── utils.py
├── viz_utils.py
├── vote_models.py
└── vote_models_predictions.py


## Environment setup

The following will setup up a conda environment suitable for running the scripts in this repo:
```
conda create -n dfc2021 "python=3.8"
conda activate dfc2021
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install tifffile matplotlib 
pip install rasterio fiona segmentation-models-pytorch

# optional steps to install a jupyter notebook kernel for this environment
pip install ipykernel
python -m ipykernel install --user --name dfc2021
```


## Baselines

The accompanying arxiv paper compares U-Nets and small fully convolutional neural networks on the task of computing high-resolution land cover change when only given _different_ low-resolution labels. We provide an implementation to reproduce the main results from this paper, and supplemental results that further compare how training a single model with multiple years of data compares to training individual models for each year. We do not include the steps for reproducing the "FCN / tile" experiment, however this experiment can be scripted using the same training/inference tools detailed below.

The 5 baseline methods we describe are:
- (NLCD difference) This simply computes change directly from the NLCD 2013 and NLCD 2016 data layers. We expect this method to perform poorly as NLCD is at a 30m resolution, does not include any changes that happened in 2017, and will systematically miss some types of changes.
- (FCN both) This is a simple [fully convolutional network (FCN)](models.py#L13) trained on pairs of both (NAIP 2013, NLCD 2013) and (NAIP 2017, NLCD 2016) imagery and labels. The trained model is then used to make indepedent predictions for the NAIP 2013 and NAIP 2017 imagery. The land cover change is calculated as a difference between the two predicted layers.
- (FCN separate) This is two FCNs -- one is trained on pairs of (NAIP 2013, NLCD 2013) imagery and labels, and the other is trained on pairs of (NAIP 2017, NLCD 2016) imagery and labels. Similar to "FCN both", the trained models are then used to make indepedent predictions for the NAIP 2013 and NAIP 2017 imagery. The land cover change is calculated as a difference between the two predicted layers.
- (U-Net both) This is the same as "FCN both", but with a [U-Net architecture](models.py#L35).
- (U-Net separate) This is the same as "FCN separate", but with a U-Net architecture.


Note that there are multiple ways to use a model that predicts NLCD labels to generate land cover change predictions for the competition. We implement two in this training/inference pipeline, however only report results for the second method in the paper:
1. (Hard assignment) Using a hard mapping between NLCD classes and reduced land cover classes. Here, each NLCD class is mapped to one of the 4 reduced land cover classes-- see the competition details or [utils.py](utils.py#L57) for this mapping. This is the default behaviour in the `inference.py` and `independent_pairs_to_predictions.py` scripts.
2. (Soft assignment) Using a soft mapping as described in the accompanying arxiv paper. To generate results with this method use the `--save_soft` flag when running `inference.py` to save output files that contain quantized per class probabilities, then use the `--soft_assignment` flag when running `independent_pairs_to_predictions.py`. *NOTE: the outputs from this process will be much larger than the first.*


## Running experiments

Each of the following subsections gives the set of commands needed to reproduce a [CodaLab](https://competitions.codalab.org/competitions/27956) submission file for the described baseline methods.


### `NLCD difference` baseline

```
conda activate dfc2021
python create_nlcd_only_baseline.py --output_dir results/nlcd_only_baseline/output/
python independent_pairs_to_predictions.py --input_dir results/nlcd_only_baseline/output/ --output_dir results/nlcd_only_baseline/submission/
cd results/nlcd_only_baseline/submission/
zip -9 -r ../nlcd_only_baseline.zip *.tif
```

### `U-Net both` baseline

```
conda activate dfc2021
python train.py --input_fn data/splits/training_set_naip_nlcd_both.csv --output_dir results/unet_both_baseline/ --save_most_recent --num_epochs 10 2> /dev/null
python inference.py --input_fn data/splits/val_inference_both.csv --model_fn results/unet_both_baseline/most_recent_model.pt --output_dir results/unet_both_baseline/output/
python independent_pairs_to_predictions.py --input_dir results/unet_both_baseline/output/ --output_dir results/unet_both_baseline/submission/
cd results/unet_both_baseline/submission/
zip -9 -r ../unet_both_baseline.zip *.tif
```

### `U-Net separate` baseline

```
conda activate dfc2021
python train.py --input_fn data/splits/training_set_naip_nlcd_2013.csv --output_dir results/unet_2013_baseline/ --save_most_recent --num_epochs 10 2> /dev/null
python train.py --input_fn data/splits/training_set_naip_nlcd_2017.csv --output_dir results/unet_2017_baseline/ --save_most_recent --num_epochs 10 2> /dev/null

python inference.py --input_fn data/splits/val_inference_2013.csv --model_fn results/unet_2013_baseline/most_recent_model.pt --output_dir results/unet_2013_baseline/output/
python inference.py --input_fn data/splits/val_inference_2017.csv --model_fn results/unet_2017_baseline/most_recent_model.pt --output_dir results/unet_2017_baseline/output/

mkdir -p results/unet_separate_baseline/output/
mkdir -p results/unet_separate_baseline/submission/
mv results/unet_2013_baseline/output/*.tif results/unet_separate_baseline/output/
mv results/unet_2017_baseline/output/*.tif results/unet_separate_baseline/output/

python independent_pairs_to_predictions.py --input_dir results/unet_separate_baseline/output/ --output_dir results/unet_separate_baseline/submission/
cd results/unet_separate_baseline/submission/
zip -9 -r ../unet_separate_baseline.zip *.tif
```

### `FCN both` baseline

```
conda activate dfc2021
python train.py --input_fn data/splits/training_set_naip_nlcd_both.csv --output_dir results/fcn_both_baseline/ --save_most_recent --model fcn --num_epochs 10 2> /dev/null
CUDA_VISIBLE_DEVICES=0 nohup python train.py --input_fn data/splits/training_set_naip_nlcd_both.csv --output_dir results/fcn_both_baseline/ --save_most_recent --model fcn --num_epochs 10 > fcn_both_baseline.log 2>&1 &
python inference.py --input_fn data/splits/val_inference_both.csv --model_fn results/fcn_both_baseline/most_recent_model.pt --output_dir results/fcn_both_baseline/output/ --model fcn
# single cls
CUDA_VISIBLE_DEVICES=0 python inference.py --input_fn data/splits/val_inference_both.csv --model_fn results/fcn_both_clean_fcnlabel_conv7/checkpoint_epoch12.pt --output_dir results/fcn_both_clean_fcnlabel_conv7/output/water_nir_ep1/ --model fcn --water_model results/water_nir/checkpoint_epoch1.pt

python independent_pairs_to_predictions.py --input_dir results/fcn_both_baseline/output/ --output_dir results/fcn_both_baseline/submission/
cd results/fcn_both_baseline/submission/
zip -9 -r ../fcn_both_baseline.zip *.tif
```

### `FCN separate` baseline

```
conda activate dfc2021
python train.py --input_fn data/splits/training_set_naip_nlcd_2013.csv --output_dir results/fcn_2013_baseline/ --save_most_recent --model fcn --num_epochs 10 2> /dev/null
python train.py --input_fn data/splits/training_set_naip_nlcd_2017.csv --output_dir results/fcn_2017_baseline/ --save_most_recent --model fcn --num_epochs 10 2> /dev/null

python inference.py --input_fn data/splits/val_inference_2013.csv --model_fn results/fcn_2013_baseline/most_recent_model.pt --output_dir results/fcn_2013_baseline/output/ --model fcn
python inference.py --input_fn data/splits/val_inference_2017.csv --model_fn results/fcn_2017_baseline/most_recent_model.pt --output_dir results/fcn_2017_baseline/output/ --model fcn

mkdir -p results/fcn_separate_baseline/output/
mkdir -p results/fcn_separate_baseline/submission/
mv results/fcn_2013_baseline/output/*.tif results/fcn_separate_baseline/output/
mv results/fcn_2017_baseline/output/*.tif results/fcn_separate_baseline/output/

python independent_pairs_to_predictions.py --input_dir results/fcn_separate_baseline/output/ --output_dir results/fcn_separate_baseline/submission/
cd results/fcn_separate_baseline/submission/
zip -9 -r ../fcn_separate_baseline.zip *.tif
```


<p align="center">
    <img src="images/fcn_unet.png" width="430"/>
</p>



If you make use of this implementation in your own project or want to refer to it in a scientific publication, **please consider referencing this GitHub repository and citing our [paper](https://arxiv.org/pdf/2101.01154.pdf)**:
```
@Article{malkinDFC2021,
  author  = {Kolya Malkin and Caleb Robinson and Nebojsa Jojic},
  title   = {High-resolution land cover change from low-resolution labels: Simple baselines for the 2021 IEEE GRSS Data Fusion Contest},
  year    = {2021},
  journal = {arXiv:2101.01154}
}
```