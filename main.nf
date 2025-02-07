#!/usr/bin/env nextflow 
/*
REFRACT Model
*/

nextflow.enable.dsl = 2

include { run_refract } from './workflows/refract'

// Define parameter for dev mode with default value of false
params.dev = false

workflow {
    run_refract(params.dev)
}