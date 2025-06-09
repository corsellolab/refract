process TRAIN_NEURAL_NETWORK_QUANTILE_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', pattern: "neural_network_quantile"
    conda params.conda_env

    // Resource requirements - neural networks may need more memory and can benefit from GPU
    cpus params.n_threads
    memory "${params.n_threads * 4} GB"

    input:
    tuple val(response_name), path(response_file), path(feature_path), path(split_dir)

    output:
    path "neural_network_quantile"

    script:
    """
    echo "run"
    mkdir -p neural_network_quantile
    python ${params.pipeline_script_dir}/train_neural_network_quantile_model.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --split_dir ${split_dir} \
        --output_dir neural_network_quantile \
        --n_threads ${params.n_threads} \
        --n_splits ${params.n_splits ?: 10} \
        --quantile ${params.quantile ?: 0.1} \
        --hidden_sizes "${params.hidden_sizes ?: '128,64,32'}" \
        --learning_rate ${params.learning_rate ?: 0.001} \
        --batch_size ${params.batch_size ?: 64} \
        --dropout_rate ${params.dropout_rate ?: 0.2} \
        --n_epochs ${params.n_epochs ?: 200} \
        --patience ${params.patience ?: 20}
    """
} 