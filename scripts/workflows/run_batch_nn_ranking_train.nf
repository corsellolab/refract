params.response_dir = '/scratch/users/nphill22/projects/corsello_lab/final_xgboost_model/data/updated_responses'
params.feature_path = '/scratch/users/nphill22/projects/corsello_lab/final_xgboost_model/data/pkl_depmap_public-22q1-305b_v24'
params.output_path = '/scratch/users/nphill22/projects/corsello_lab/final_xgboost_model/output_primary_screen'
params.dev = false

process train_nn_ranking {
    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }
    cpus { 8 * task.attempt }
    memory { 32.GB * task.attempt }
    time { 1.hour * task.attempt }
    executor 'slurm'
    conda '/home/groups/dkurtz/tools/conda/miniconda3/envs/corlab'
    publishDir "${params.output_path}/{response_path.baseName}", mode: 'copy'

    input:
    path response_path
    path feature_path

    output:
    path "${response_path.baseName}"

    script:
    """
    mkdir -p ${params.output_path}/${response_path.baseName}
    python /scratch/users/nphill22/projects/corsello_lab/final_xgboost_model/refract/scripts/run_training.py --response_path ${response_path} --feature_path ${feature_path} --output_dir ${response_path.baseName}
    """
}

workflow {
    response_files = Channel.fromPath("${params.response_dir}/*.csv")
    if (params.dev) {
        response_files = response_files.take(3)
    }
    response_files
        .map { file("${it}") }
        .set { ch_response_files }
    train_nn_ranking(ch_response_files, file(params.feature_path))
}
