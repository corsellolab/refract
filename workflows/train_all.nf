#!/usr/bin/env nextflow 

include { SELECT_TOP_FEATURES } from "../modules/split_data.nf"

workflow train_all {
    take:
    response_ch
    feature_path

    main:
    // run split data
    SELECT_TOP_FEATURES(response_ch, feature_path)
}