process SELECT_FEATURES {
    publishDir params.split_output_dir, mode: 'copy'

    input:
    path feature_file
    path response_file

    output:
    path "split_data"

    script:
    """
    mkdir -p split_data
    python ${projectDir}/refract/scripts/select_features.py \
        --feature_file ${feature_file} \
        --response_file ${response_file} \
        --output_dir split_data \
        --n_splits ${params.n_splits} \
        --feature_fraction ${params.feature_fraction}
    """
}