#!/usr/bin/env Rscript

library(tidyverse)
library(data.table)
library(ranger)
library(reshape2)

cat("Starting Random Forest prediction analysis...\n")

# Command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript RF_predict.R <feature_file> <response_file> <fold_assignment_file> <top_features_file> <output_dir>")
}

feature_file <- args[1]
response_file <- args[2]
fold_assignment_file <- args[3]
top_features_file <- args[4]
output_dir <- args[5]

cat(sprintf("Parameters:\n  Feature file: %s\n  Response file: %s\n  Fold assignment file: %s\n  Top features file: %s\n  Output directory: %s\n\n",
            feature_file, response_file, fold_assignment_file, top_features_file, output_dir))

# Create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Read input files
cat("Reading input files...\n")
feature_data <- as.data.frame(readRDS(feature_file))  # Convert to data.frame immediately
response_data <- as.data.frame(fread(response_file))
fold_assignments <- as.data.frame(fread(fold_assignment_file))
top_features <- as.data.frame(fread(top_features_file))

cat(sprintf("Loaded:\n  %d features for %d samples\n  %d response measurements\n  %d fold assignments\n  %d top features\n",
            ncol(feature_data)-1, nrow(feature_data),
            nrow(response_data),
            nrow(fold_assignments),
            nrow(top_features)))

# Join response and feature data
cat("Joining response and feature data...\n")
joined_data <- response_data %>%
  left_join(feature_data, by = c("depmap_id" = "ModelID"))
joined_data <- as.data.frame(joined_data)  # Ensure it's a data.frame

# Initialize results storage
all_predictions <- data.frame()
all_importances <- data.frame()
unique_folds <- unique(fold_assignments$fold_idx)

# Function to train RF model and get predictions
train_rf_model <- function(X_train, X_test, y_train, features) {
    # Prepare training data
    train_data <- as.data.frame(X_train[, c(features), drop=FALSE])
    
    # Sanitize column names
    names(train_data) <- make.names(names(train_data), unique = TRUE)
    train_data$y_label <- y_train
    
    # Handle missing values
    train_data[is.na(train_data)] <- 0  # Simpler NA handling
    
    # Train RF model
    rf_model <- ranger(y_label ~ ., 
                      data = train_data,
                      importance = "impurity",
                      num.trees = 500)
    
    # Prepare test data
    test_data <- as.data.frame(X_test[, c(features), drop=FALSE])
    names(test_data) <- make.names(names(test_data), unique = TRUE)  # Apply same sanitization
    test_data[is.na(test_data)] <- 0  # Simpler NA handling
    
    # Get predictions
    predictions <- predict(rf_model, test_data)$predictions
    
    # Get feature importance
    importance <- data.frame(
        feature = features,  # Use original feature names
        importance = rf_model$variable.importance / sum(rf_model$variable.importance)
    )
    
    return(list(predictions = predictions, importance = importance))
}

# Iterate over folds
cat("\nProcessing folds...\n")
for (fold in unique_folds) {
    cat(sprintf("Processing fold %d...\n", fold))
    
    # Split data into train and test
    test_samples <- fold_assignments$depmap_id[fold_assignments$fold_idx == fold]
    train_samples <- fold_assignments$depmap_id[fold_assignments$fold_idx != fold]
    
    # Get features for this fold
    fold_features <- as.character(top_features$feature_name[top_features$fold_idx == fold])
    
    # Ensure all features exist in the data
    fold_features <- intersect(fold_features, colnames(joined_data))
    
    if (length(fold_features) == 0) {
        cat(sprintf("Warning: No valid features found for fold %d\n", fold))
        next
    }
    
    # Prepare X and y
    X <- joined_data[, c("depmap_id", fold_features)]
    y <- joined_data$LFC
    
    X_train <- X[joined_data$depmap_id %in% train_samples, ]
    X_test <- X[joined_data$depmap_id %in% test_samples, ]
    y_train <- y[joined_data$depmap_id %in% train_samples]
    y_test <- y[joined_data$depmap_id %in% test_samples]
    
    # Train model and get predictions
    results <- train_rf_model(X_train, X_test, y_train, fold_features)
    
    # Store predictions
    fold_predictions <- data.frame(
        depmap_id = test_samples,
        fold = fold,
        predicted = results$predictions,
        actual = y_test
    )
    all_predictions <- rbind(all_predictions, fold_predictions)
    
    # Store feature importance
    fold_importance <- results$importance
    fold_importance$fold <- fold
    all_importances <- rbind(all_importances, fold_importance)
}

# Calculate model performance metrics
cat("\nCalculating model performance metrics...\n")
model_metrics <- data.frame(
    pearson_correlation = cor(all_predictions$predicted, all_predictions$actual, use="complete.obs"),
    r_squared = 1 - sum((all_predictions$actual - all_predictions$predicted)^2, na.rm=TRUE) / 
                sum((all_predictions$actual - mean(all_predictions$actual, na.rm=TRUE))^2, na.rm=TRUE),
    mse = mean((all_predictions$actual - all_predictions$predicted)^2, na.rm=TRUE)
)
model_metrics$rmse <- sqrt(model_metrics$mse)
model_metrics$mae <- mean(abs(all_predictions$actual - all_predictions$predicted), na.rm=TRUE)

# Summarize feature importance across folds
feature_summary <- aggregate(
    importance ~ feature, 
    data = all_importances, 
    FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x))
)
feature_summary <- data.frame(
    feature = feature_summary$feature,
    mean_importance = feature_summary$importance[,1],
    sd_importance = feature_summary$importance[,2],
    n_folds = feature_summary$importance[,3]
)
# Sort by descending mean importance
feature_summary <- feature_summary[order(-feature_summary$mean_importance), ]

cat(sprintf("  Sorted %d features by importance...\n", nrow(feature_summary)))

# Save results
cat("\nSaving results...\n")
write.csv(all_predictions, file.path(output_dir, "rf_predictions.csv"), row.names = FALSE)
write.csv(model_metrics, file.path(output_dir, "rf_model_metrics.csv"), row.names = FALSE)
write.csv(feature_summary, file.path(output_dir, "rf_feature_importance.csv"), row.names = FALSE)

cat("\nRandom Forest analysis complete!\n") 