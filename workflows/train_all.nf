#!/usr/bin/env nextflow 

include { SPLIT_DATA } from "../modules/split_data.nf"
include { TRAIN_XGBOOST_MODEL } from "../modules/train_xgboost_model.nf"

workflow train_all {
    take:
    response_ch
    feature_path

    main:
    // run split data
    SPLIT_DATA(response_ch, feature_path)

    // train xgboost model
    TRAIN_XGBOOST_MODEL(response_ch, feature_path, SPLIT_DATA.out.data_split)
}