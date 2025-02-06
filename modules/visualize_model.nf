process VISUALIZE_MODEL {
    publishDir params.visualization_dir, mode: 'copy'

    input:
    path model_dir

    output:
    path "${model_dir.baseName}_viz"

    script:
    """
    mkdir -p ${model_dir.baseName}_viz
    python ${params.pipeline_script_dir}/visualize_model.py \
        --model_dir ${model_dir} \
        --output_dir ${model_dir.baseName}_viz
    """
} 