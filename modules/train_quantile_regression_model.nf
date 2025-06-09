process TRAIN_QUANTILE_REGRESSION_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', pattern: "quantile_regression"
    conda params.conda_env

    // Resource requirements
    cpus params.n_threads
    memory "${params.n_threads * 2} GB"

    input:
    tuple val(response_name), path(response_file), path(feature_path), path(split_dir)

    output:
    path "quantile_regression"

    script:
    """
    echo "run"
    mkdir -p quantile_regression
    python ${params.pipeline_script_dir}/train_quantile_regression_model.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --split_dir ${split_dir} \
        --output_dir quantile_regression \
        --n_threads ${params.n_threads} \
        --n_splits ${params.n_splits ?: 10} \
        --quantile ${params.quantile ?: 0.1} \
        --alphas "${params.alphas_qr ?: '0.0,0.001,0.01,0.1,1.0,10.0'}" \
        --solvers "${params.solvers ?: 'highs,interior-point'}"
    """
} 