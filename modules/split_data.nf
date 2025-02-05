process SELECT_FEATURES_REFRACT {
    publishDir params.output_dir, mode: 'copy'

    input:
    tuple val(response_name), path(response_file)

    output:
    path "${response_name}"

    script:    
    """
    mkdir -p ${response_name}
    python ${params.pipeline_script_dir}/select_features.py \
        --feature_file ${params.feature_path} \
        --response_file ${response_file} \
        --output_dir ${response_name} \
        --n_splits ${params.n_splits} \
        --feature_fraction ${params.feature_fraction}
    """
}