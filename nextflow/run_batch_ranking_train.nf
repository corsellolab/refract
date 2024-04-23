params.response_dir = '/scratch/users/nphill22/projects/corsello_lab/20240313_prism_final_reruns/data/features/responses'
params.feature_path = '/scratch/users/nphill22/projects/corsello_lab/20240313_prism_final_reruns/data/features/v3_updated_x-all.pkl'
params.output_path = '/scratch/users/nphill22/projects/corsello_lab/20240313_prism_final_reruns/data/rf_output'
params.dev = false

process train_rank {
    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' }
    cpus { 8 * task.attempt }
    memory { 16.GB * task.attempt }
    time { 2.hour * task.attempt }
    executor 'slurm'
    conda '/scratch/users/nphill22/temp/miniconda3'
    publishDir "${params.output_path}", mode: 'copy'

    input:
    tuple val(drug_name), path(response_path)

    output:
    path("${drug_name}")

    script:
    """
    python /scratch/users/nphill22/projects/corsello_lab/20240313_prism_final_reruns/refract/run_training.py --response_path ${response_path} --feature_path ${params.feature_path} --output_dir ${drug_name}
    """
}

workflow {
response_files = Channel.fromPath("${params.response_dir}/*.csv")
    if (params.dev) {
        response_files = response_files.take(3)
    }
    response_files
        .map { file_name -> tuple(file_name.baseName, file("${file_name}")) }
        .set { ch_response_files }
    train_rank(ch_response_files)
}
