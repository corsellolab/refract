params.response_dir = '/scratch/users/nphill22/projects/corsello_lab/20240706_retrain_feature_select/data/responses'
params.feature_path = " /scratch/users/nphill22/projects/corsello_lab/20240706_retrain_feature_select/data/processed_data/x-all.pkl"
params.output_path = "/scratch/users/nphill22/projects/corsello_lab/20240706_retrain_feature_select/outputs/cv_mid_size_test"
params.neighborhood_json = "/scratch/users/nphill22/projects/corsello_lab/20240706_retrain_feature_select/refract/notebooks/20240706_feature_selection/neighbors.json"
params.drug_list_file = "/scratch/users/nphill22/projects/corsello_lab/20240706_retrain_feature_select/refract/notebooks/20240706_feature_selection/sampled_drugs.txt"
params.dev = false


process train_rank {
    errorStrategy { task.attempt < 2 ? 'retry' : 'ignore' }
    cpus { 8 * task.attempt }
    memory { 32.GB * task.attempt }
    time { 2.hour * task.attempt }
    executor 'slurm'
    conda '/scratch/users/nphill22/conda_installs/miniconda/envs/prism'
    publishDir "${params.output_path}", mode: 'copy'

    input:
    val(drug_name)

    output:
    path("${drug_name}")

    script:
    """
    python /scratch/users/nphill22/projects/corsello_lab/20240706_retrain_feature_select/refract/run_training.py --drug_name ${drug_name} --response_dir ${params.response_dir} --feature_path ${params.feature_path} --output_dir ${drug_name} --neighborhood_json ${params.neighborhood_json}
    """
}

workflow {
    // Read drug names from the file
    Channel
        .fromPath(params.drug_list_file)
        .splitText()
        .map { it.trim() }
        .set { drug_names }

    if (params.dev) {
        drug_names = drug_names.take(3)
    }

    train_rank(drug_names)

}
