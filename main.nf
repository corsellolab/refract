#!/usr/bin/env nextflow 
/*
REFRACT Model
*/

nextflow.enable.dsl = 2
include { train_all } from './workflows/train_all'

// Default parameters
params.dev = false
params.script_dir = "${projectDir}/scripts"
params.seed = 42
params.n_threads = 8

// Function to create channel from directory files
def create_file_channel(directory) {
    Channel
        .fromPath("${directory}/*")
        .filter { it.isFile() }
        .map { file -> 
            def fileName = file.getBaseName()
            [fileName, file]
        }
}

workflow {
    // Create file channel from response directory
    response_ch = create_file_channel(params.response_dir)

    // Limit to 2 files if in dev mode
    if (params.dev) {
        response_ch = response_ch.take(100)
    }

    // Call train_all workflow
    train_all(response_ch, params.feature_path_pkl)
}