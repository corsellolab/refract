params.response_dir = '/scratch/users/nphill22/projects/corsello_lab/20230911_follow_up/data/primary_23q2'
params.feature_path = '/scratch/users/nphill22/projects/corsello_lab/20230911_follow_up/data/features/v2_x-all.pkl'
params.output_path = '/scratch/users/nphill22/projects/corsello_lab/20230911_follow_up/data/output'
params.dev = false

process train_rank {
    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' }
    cpus { 12 * task.attempt }
    memory { 48.GB * task.attempt }
    time { 1.hour * task.attempt }
    executor 'slurm'
    conda '/home/groups/dkurtz/tools/conda/miniconda3/envs/corlab'
    publishDir "${params.output_path}", mode: 'copy'

    input:
    tuple val(drug_name), path(response_path), val(sample_frac)

    output:
    path("${drug_name}_${sample_frac}")

    script:
    """
    python /scratch/users/nphill22/projects/corsello_lab/20230911_follow_up/refract/run_training.py --response_path ${response_path} --feature_path ${params.feature_path} --output_dir ${drug_name}_${sample_frac} --feature_fraction ${sample_frac}
    """
}

workflow {
    csvChannel = Channel.fromPath("/scratch/users/nphill22/projects/corsello_lab/20230911_follow_up/data/optimize_frac.csv")
        .splitCsv(header: true, sep: ',')
        .map { row -> [row.drug, row.path, row.frac] }
    
    train_rank(csvChannel)
}
