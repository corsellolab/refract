#!/usr/bin/env nextflow 

include { SPLIT_DATA } from "../modules/split_data.nf"
include { TRAIN_XGBOOST_MODEL } from "../modules/train_xgboost_model.nf"
include { TRAIN_LINEAR_MODEL } from "../modules/train_linear_model.nf"

workflow train_all {
    take:
    response_ch
    feature_path

    main:
    // run split data
    SPLIT_DATA(response_ch, feature_path)

    // train xgboost model
    TRAIN_XGBOOST_MODEL(SPLIT_DATA.out)

    // train linear model
    TRAIN_LINEAR_MODEL(SPLIT_DATA.out)
}