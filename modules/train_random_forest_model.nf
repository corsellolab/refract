process TRAIN_RANDOM_FOREST_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', pattern: "random_forest"
    conda params.conda_env

    // Resource requirements
    cpus params.n_threads
    memory "${params.n_threads * 2} GB"

    input:
    tuple val(response_name), path(response_file), path(feature_path), path(split_dir)

    output:
    path "random_forest"

    script:
    """
    mkdir -p random_forest
    python ${params.pipeline_script_dir}/train_random_forest_model.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --split_dir ${split_dir} \
        --output_dir random_forest \
        --n_threads ${params.n_threads} \
        --n_splits ${params.n_splits ?: 10} \
        --n_estimators "${params.n_estimators ?: '100'}" \
        --max_depth "${params.max_depth ?: 'None'}" \
        --min_samples_split "${params.min_samples_split ?: '2'}" \
        --min_samples_leaf "${params.min_samples_leaf ?: '1'}" \
        --max_features "${params.max_features ?: '1.0'}"
    """
} 