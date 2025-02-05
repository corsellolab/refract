#!/usr/bin/env nextflow 
/*
REFRACT Model
*/

nextflow.enable.dsl = 2

include { train_refract } from './workflows/train_refract'

// Define parameter for dev mode with default value of false
params.dev = false

workflow {
    train_refract(params.dev)
}