process SELECT_FEATURES_REFRACT {
    publishDir params.data_split_dir, mode: 'copy', pattern: "${response_name}"

    input:
    tuple val(response_name), path(response_file)
    path(feature_path)

    output:
    tuple val(response_name), path(response_file), path(feature_path), path("${response_name}")

    script:    
    """
    mkdir -p ${response_name}
    python ${params.pipeline_script_dir}/select_features.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --output_dir ${response_name} \
        --n_splits ${params.n_splits} \
        --feature_fraction ${params.feature_fraction} \
        --train_val_split ${params.train_val_split}
    """
}