# commands to run different workflows

# subset primary
nextflow run main.nf -resume \
--output_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/output/subset/primary \
--response_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/responses/subset/primary_screen \
--feature_path_pkl /drive3/nphill22/projects/corsello_lab/20250525_manuscript/sherlock_copy/data/processed_features/x-all.pkl

# full primary
nextflow run main.nf -resume \
--output_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/output/full/primary \
--response_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/responses/full/primary_screen \
--feature_path_pkl /drive3/nphill22/projects/corsello_lab/20250525_manuscript/sherlock_copy/data/processed_features/x-all.pkl

# full rep1m
nextflow run main.nf -resume \
--output_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/output/full/rep1m \
--response_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/responses/full/rep1m \
--feature_path_pkl /drive3/nphill22/projects/corsello_lab/20250525_manuscript/sherlock_copy/data/processed_features/x-all.pkl

nextflow run main.nf -resume \
--output_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/output/full/secondary_screen \
--response_dir /drive3/nphill22/projects/corsello_lab/20250525_manuscript/data/responses/full/secondary_screen \
--feature_path_pkl /drive3/nphill22/projects/corsello_lab/20250525_manuscript/sherlock_copy/data/processed_features/x-all.pkl
