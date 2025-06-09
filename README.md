# Refract

Ranking models for analysis of PRISM multiplexed cell-line datasets.

This package provides multiple machine learning approaches for drug response prediction, including:
- XGBoost regression
- Linear (Lasso) regression  
- Random Forest regression
- Quantile regression
- Neural Network quantile regression

## Setup
Setup a conda environment: 
```
conda env create -f environment.yml
conda activate refract
```

## Usage
Baseline model training: 
```
python scripts/run_training.py \
    --response_path <path_to_response_csv> \
    --feature_path <path_to_feature_pkl_file> \
    --output_dir <output_directory>
```
