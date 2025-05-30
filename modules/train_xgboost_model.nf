process TRAIN_XGBOOST_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', pattern: "xgboost_model"
    conda params.conda_env

    // Resource requirements
    cpus params.n_threads
    memory "${params.n_threads * 2} GB"

    input:
    tuple val(response_name), path(response_file), path(feature_path), path(split_dir)

    output:
    path "xgboost_model"

    script:
    """
    mkdir -p xgboost_model
    python ${params.pipeline_script_dir}/train_xgboost_model.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --split_dir ${split_dir} \
        --output_dir xgboost_model \
        --n_threads ${params.n_threads} \
        --num_rounds ${params.num_rounds ?: 1000} \
        --early_stopping_rounds ${params.early_stopping_rounds ?: 50} \
        --n_splits ${params.n_splits ?: 10} \
        --eta ${params.eta ?: 0.01} \
        --max_depth ${params.max_depth ?: 6} \
        --subsample ${params.subsample ?: 0.8} \
        --colsample_bytree ${params.colsample_bytree ?: 0.8} \
        --lambda_reg ${params.lambda_reg ?: 1.0} \
        --alpha_reg ${params.alpha_reg ?: 0.1}
    """
}