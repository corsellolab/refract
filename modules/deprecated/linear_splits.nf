process LINEAR_SPLITS {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', saveAs: { filename ->
        filename.equals("fold_assignment.csv") ? "linear_splits/fold_assignment.csv" :
        filename.equals("top_features.csv") ? "linear_splits/top_features.csv" : null
    }

    input:
    tuple val(response_name), 
          path(response_file), 
          path(feature_file),
          val(output_dir)

    output:
    tuple val(response_name), 
          path(response_file), 
          path("fold_assignment.csv"), 
          path("top_features.csv"),
          val(response_name)

    script:
    """
    Rscript ${params.script_dir}/linear_splits.R \\
        ${response_file} \\
        ${feature_file} \\
        . \\
        ${params.seed} \\
        ${params.n_threads}
    """
} 