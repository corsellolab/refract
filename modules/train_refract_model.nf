process TRAIN_REFRACT_MODEL {
    publishDir params.refract_model_dir, mode: 'copy'

    input:
    tuple val(response_name), path(response_file), path(feature_file), path(split_dir)

    output:
    path "${response_name}_model"

    script:
    """
    mkdir -p ${response_name}_model
    python ${params.pipeline_script_dir}/train_xgboost_model.py \
        --feature_file ${feature_file} \
        --response_file ${response_file} \
        --split_dir ${split_dir} \
        --output_dir ${response_name}_model \
        --n_threads ${params.n_threads}
    """
}
