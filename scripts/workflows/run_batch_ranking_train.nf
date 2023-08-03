params.response_dir = '/scratch/users/nphill22/projects/corsello_lab/final_xgboost_model/data/updated_responses'
params.feature_path = '/scratch/users/nphill22/projects/corsello_lab/final_xgboost_model/data/pkl_depmap_public-22q1-305b_v24/x-all.pkl'
params.output_path = '/scratch/users/nphill22/projects/corsello_lab/xgboost_sample_model_lastone/output'
params.dev = false

process train_nn_ranking {
    errorStrategy { task.attempt <= 2 ? 'retry' : 'ignore' }
    cpus { 8 }
    memory { 32.GB }
    time { 1.hour * task.attempt }
    executor 'slurm'
    conda '/home/groups/dkurtz/tools/conda/miniconda3/envs/corlab'
    publishDir "${params.output_path}", mode: 'copy'

    input:
    path response_path
    path feature_path

    output:
    path "${response_path.baseName}"

    script:
    """
    python /scratch/users/nphill22/projects/corsello_lab/xgboost_sample_model_lastone/refract/scripts/run_training.py --response_path ${response_path} --feature_path ${feature_path} --output_dir ${response_path.baseName}
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
