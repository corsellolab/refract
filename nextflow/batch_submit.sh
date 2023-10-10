#!/bin/bash
#SBATCH --job-name=nextflow_batch_run
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=dkurtz,owners,normal

/home/users/nphill22/nextflow run /scratch/users/nphill22/projects/corsello_lab/20231006_permutation_analysis/refract/nextflow/run_batch_ranking_train.nf -resume
