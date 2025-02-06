include { SELECT_FEATURES_REFRACT } from '../modules/split_data'
include { RUN_MTS_BASELINE } from '../modules/mts_baseline'
include { TRAIN_REFRACT_MODEL } from '../modules/train_refract_model'
include { VISUALIZE_MODEL } from '../modules/visualize_model'

workflow run_refract {
    take:
        dev_mode

    main:
        // Get all response CSV files
        response_files = Channel
            .fromPath("${params.response_dir}/*.csv")
            .take(dev_mode ? 2 : -1) // If dev_mode is true, take only 2 files

        // Process each response file
        response_files.map { file -> 
            def response_name = file.name.replace('.csv','')
            return tuple(response_name, file)
        }.set { named_responses }

        // Run feature selection for each response using the module
        SELECT_FEATURES_REFRACT(
            named_responses,
            params.feature_path
        )

        // Run MTS baseline using the same splits
        RUN_MTS_BASELINE(
            SELECT_FEATURES_REFRACT.out.map { name, response, _feature_path, splits -> tuple(name, response, splits) },
            params.feature_path_rds  // Use the RDS file for R script
        )

        // Train models using the selected features
        TRAIN_REFRACT_MODEL(
            SELECT_FEATURES_REFRACT.out
        )

        // Add visualization step
        VISUALIZE_MODEL(
            TRAIN_REFRACT_MODEL.out
        )
}
