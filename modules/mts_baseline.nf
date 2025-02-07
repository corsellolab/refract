process RUN_MTS_BASELINE {
    publishDir "${params.mts_baseline_dir}/${response_name}", mode: 'copy'
    
    input:
    tuple val(response_name), path(response_file), path(split_dir)
    path feature_file

    output:
    tuple val(response_name), 
          path("pred_true.csv"), 
          path("RF_table.csv"), 
          path("Model_table.csv")

    script:
    """
    Rscript ${params.pipeline_script_dir}/MTS_Analysis.R \
        ${feature_file} \
        ${response_file} \
        . \
        ${split_dir}
    """
} 