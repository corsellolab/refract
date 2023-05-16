# Refract
## A python implementation of MTS analysis

Temporary working title. Extending the PRISM motif.

## Setup
Setup a conda environment: 
```
conda env create -f environment.yml
conda activate corlab
```

## Usage
Baseline model training: 
```
python scripts/runCVRFTrain.py \
    --feature_dir <path_to_feature_directory> \
    --response_path <path_to_response_csv> \
    --output_dir <path_to_output_root> \
    --config_path <path_to_config_json> (optional)
```

See `refract/utils.py` for optional configurations. 
