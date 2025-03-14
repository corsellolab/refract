process LINEAR_MODEL {
    publishDir "${params.output_dir}/${response_name}", mode: 'copy', saveAs: { filename ->
        filename.equals("linear_predictions.csv") ? "linear_model/linear_predictions.csv" :
        filename.equals("linear_model_metrics.csv") ? "linear_model/linear_model_metrics.csv" :
        filename.equals("linear_feature_importance.csv") ? "linear_model/linear_feature_importance.csv" : null
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
          path("linear_predictions.csv"),
          path("linear_model_metrics.csv"),
          path("linear_feature_importance.csv"),
          val(response_name)

    script:
    """
    echo "linear model"
    Rscript ${params.script_dir}/lin_predict.R \\
        ${feature_file} \\
        ${response_file} \\
        ${fold_assignment} \\
        ${top_features} \\
        .
    """
} 