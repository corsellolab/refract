process RF_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', saveAs: { filename ->
        filename.equals("rf_predictions.csv") ? "rf_model/rf_predictions.csv" :
        filename.equals("rf_model_metrics.csv") ? "rf_model/rf_model_metrics.csv" :
        filename.equals("rf_feature_importance.csv") ? "rf_model/rf_feature_importance.csv" : null
    }

    input:
    tuple val(response_name), 
          path(response_file), 
          path(fold_assignment), 
          path(top_features),
          val(response_subdir)
    path feature_file

    output:
    tuple val(response_name),
          path("rf_predictions.csv"),
          path("rf_model_metrics.csv"),
          path("rf_feature_importance.csv"),
          val(response_name)

    script:
    """
    Rscript ${params.script_dir}/RF_predict.R \\
        ${feature_file} \\
        ${response_file} \\
        ${fold_assignment} \\
        ${top_features} \\
        .
    """
} 