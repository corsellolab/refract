include { SELECT_FEATURES_REFRACT } from '../modules/split_data'

workflow train_refract {
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
            named_responses
        )
}
