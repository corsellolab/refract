#!/usr/bin/env nextflow 

include { SPLIT_DATA } from "../modules/split_data.nf"
include { TRAIN_XGBOOST_MODEL } from "../modules/train_xgboost_model.nf"
include { TRAIN_LINEAR_MODEL } from "../modules/train_linear_model.nf"
include { TRAIN_RANDOM_FOREST_MODEL } from "../modules/train_random_forest_model.nf"
include { TRAIN_QUANTILE_REGRESSION_MODEL } from "../modules/train_quantile_regression_model.nf"
include { TRAIN_NEURAL_NETWORK_QUANTILE_MODEL } from "../modules/train_neural_network_quantile_model.nf"

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

    // train random forest model
    TRAIN_RANDOM_FOREST_MODEL(SPLIT_DATA.out)

    // train quantile regression model
    TRAIN_QUANTILE_REGRESSION_MODEL(SPLIT_DATA.out)

    // train neural network quantile regression model
    TRAIN_NEURAL_NETWORK_QUANTILE_MODEL(SPLIT_DATA.out)
}