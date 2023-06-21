#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.run_manifest = "/scratch/users/nphill22/projects/corsello_lab/configure_ranking_NN_runs/paths_for_ranking_model.csv"
params.output_dir = "xgboost_output"
params.dev = false

process train_nn_ranking {
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'ignore' }
    cpus { 4 * task.attempt }
    memory { 16.GB * task.attempt }
    time { 1.hour * task.attempt }
    maxRetries 3
    executor 'slurm'
    conda '/home/groups/dkurtz/tools/conda/miniconda3/envs/corlab'
    publishDir params.output_dir, mode: 'copy'

    input:
        tuple val(drug_name), file(response_path), file(feature_path), file(feature_importance_path)

    output:
        path("${drug_name}")

    script:
    """
    mkdir -p ${drug_name}
    python /scratch/users/nphill22/projects/corsello_lab/configure_ranking_NN_runs/refract/scripts/runXGBoostRankingTrain.py --response_path ${response_path} --feature_path ${feature_path} --feature_importance_path ${feature_importance_path} --output_dir ${drug_name}
    """
}

workflow {
    Channel
        .fromPath( params.run_manifest )
        .splitCsv(header: true, sep:',')
        .map { row -> tuple(row.drug_name, file(row.response_path), file(row.feature_path), file(row.feature_importance_path)) }
        .take(params.dev ? 2 : -1)
        .set { run_manifest_ch }

    train_nn_ranking(run_manifest_ch)
}
