#!/usr/bin/env nextflow 
/*
REFRACT Model
*/

nextflow.enable.dsl = 2

include { run_refract } from './workflows/refract'
include { run_baselines } from './workflows/baselines'

// Default parameters
params.dev = false
params.run_baselines = true
params.sample_sheet = null
params.script_dir = "${projectDir}/scripts"
params.seed = 42
params.n_threads = 8

// Function to read sample sheet
def create_sample_sheet_channel() {
    if (params.sample_sheet) {
        Channel
            .fromPath(params.sample_sheet)
            .splitCsv(header: true, sep: ',')
            .map { row -> 
                [ row.response_name, 
                  file(row.response_file), 
                  file(row.feature_file),
                  row.response_name ]
            }
    } else {
        Channel.empty()
    }
}

workflow {
    if (params.run_baselines) {
        // Run baseline models
        sample_sheet_ch = create_sample_sheet_channel()
        run_baselines(sample_sheet_ch)
    } else {
        // Run main REFRACT pipeline
        run_refract(params.dev)
    }
}