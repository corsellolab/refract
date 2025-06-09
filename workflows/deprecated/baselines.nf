#!/usr/bin/env nextflow

include { LINEAR_SPLITS } from '../modules/linear_splits'
include { LINEAR_MODEL } from '../modules/linear_model'
include { RF_MODEL } from '../modules/rf_model'

workflow run_baselines {
    take:
    sample_sheet_ch  // channel containing sample sheet rows: [response_name, response_file, feature_file]

    main:
    // Run linear splits
    LINEAR_SPLITS(sample_sheet_ch)

    // Run linear model
    LINEAR_MODEL(LINEAR_SPLITS.out, sample_sheet_ch.map { it[2] })

    // Run random forest model
    RF_MODEL(LINEAR_SPLITS.out, sample_sheet_ch.map { it[2] })

    emit:
    linear_results = LINEAR_MODEL.out
    rf_results = RF_MODEL.out
} 