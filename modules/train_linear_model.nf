process TRAIN_LINEAR_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', pattern: "linear_model"
    conda params.conda_env

    // Resource requirements
    cpus params.n_threads
    memory "${params.n_threads * 2} GB"

    input:
    tuple val(response_name), path(response_file), path(feature_path), path(split_dir)

    output:
    path "linear_model"

    script:
    """
    mkdir -p linear_model
    python ${params.pipeline_script_dir}/train_linear_model.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --split_dir ${split_dir} \
        --output_dir linear_model \
        --n_threads ${params.n_threads} \
        --n_splits ${params.n_splits ?: 10} \
        --max_iter ${params.max_iter ?: 10000} \
        --alphas "${params.alphas ?: '0.001,0.01,0.1,1.0,10.0,100.0'}"
    """
} 