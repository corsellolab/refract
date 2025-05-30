process SPLIT_DATA {
    //publishDir "${params.output_dir}", pattern: "${params.data_split_dir}"
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', pattern: "${params.data_split_dir}"
    conda params.conda_env
    
    // Resource requirements
    cpus 1
    memory '8 GB'

    input:
    tuple val(response_name), path(response_file)
    path(feature_path)

    output:
    tuple val(response_name), path(response_file), path(feature_path), path("${params.data_split_dir}")

    script:
    """
    python ${params.pipeline_script_dir}/prepare_splits.py \
        --feature_file ${feature_path} \
        --response_file ${response_file} \
        --output_dir ${params.data_split_dir} \
        --n_splits ${params.n_splits} \
        --n_features ${params.n_features} \
        --train_val_split ${params.train_val_split}
    """
}