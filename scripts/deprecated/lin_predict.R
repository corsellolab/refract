#!/usr/bin/env Rscript

# Load required libraries
library(tidyverse)
library(data.table)
library(glmnet)
library(Matrix)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript lin_predict.R <feature_file> <response_file> <fold_assignment_file> <top_features_file> <output_dir>")
}

feature_file <- args[1]
response_file <- args[2]
fold_assignment_file <- args[3]
top_features_file <- args[4]
output_dir <- args[5]

cat("\nStarting Linear Model prediction analysis...")
cat("\nParameters:")
cat("\n  Feature file:", feature_file)
cat("\n  Response file:", response_file)
cat("\n  Fold assignment file:", fold_assignment_file)
cat("\n  Top features file:", top_features_file)
cat("\n  Output directory:", output_dir)
cat("\n")

# Read input files
cat("\nReading input files...")
features <- readRDS(feature_file)
response <- fread(response_file)
fold_assignment <- fread(fold_assignment_file)
top_features <- fread(top_features_file)

cat("\nLoaded:")
cat("\n  ", ncol(features) - 1, "features for", nrow(features), "samples")
cat("\n  ", nrow(response), "response measurements")
cat("\n  ", nrow(fold_assignment), "fold assignments")
cat("\n  ", nrow(top_features), "top features")

# Join response and feature data
cat("\nJoining response and feature data...")
data <- as.data.frame(features) %>%
  rename(depmap_id = ModelID) %>%  # Rename ModelID to depmap_id
  inner_join(response, by = "depmap_id") %>%
  inner_join(fold_assignment, by = "depmap_id") %>%
  filter(!is.na(LFC))  # Remove rows with missing LFC values

cat("\nAfter removing missing values:")
cat("\n  ", nrow(data), "samples remaining")

# Function to standardize features
standardize_features <- function(data, feature_cols) {
  data_std <- data
  for (col in feature_cols) {
    if (is.numeric(data[[col]])) {
      mean_val <- mean(data[[col]], na.rm = TRUE)
      sd_val <- sd(data[[col]], na.rm = TRUE)
      if (sd_val > 0) {
        data_std[[col]] <- (data[[col]] - mean_val) / sd_val
      }
    }
  }
  return(data_std)
}

# Function to train model and get predictions
train_ridge_model <- function(train_data, test_data, feature_cols, response_col) {
  # Prepare matrices for glmnet
  x_train <- as.matrix(train_data[, feature_cols])
  y_train <- train_data[[response_col]]
  x_test <- as.matrix(test_data[, feature_cols])
  
  # Handle missing values
  x_train[is.na(x_train)] <- 0
  x_test[is.na(x_test)] <- 0
  
  # Standardize features
  x_train_std <- scale(x_train)
  center <- attr(x_train_std, "scaled:center")
  scale <- attr(x_train_std, "scaled:scale")
  x_test_std <- scale(x_test, center = center, scale = scale)
  
  # Find optimal lambda using cross-validation
  cv_fit <- cv.glmnet(x_train_std, y_train, alpha = 0)
  
  # Fit ridge model with optimal lambda
  model <- glmnet(x_train_std, y_train, alpha = 0, lambda = cv_fit$lambda.min)
  
  # Get predictions
  predictions <- predict(model, x_test_std)
  
  # Get standardized coefficients
  coef_df <- data.frame(
    feature = feature_cols,
    coefficient = as.vector(coef(model)[-1]),  # Exclude intercept
    stringsAsFactors = FALSE
  ) %>%
    filter(!is.na(coefficient))
  
  return(list(
    predictions = predictions,
    coefficients = coef_df,
    rmse = sqrt(mean((predictions - test_data[[response_col]])^2)),
    mae = mean(abs(predictions - test_data[[response_col]]))
  ))
}

# Initialize results storage
all_predictions <- data.frame()
all_coefficients <- data.frame()
fold_metrics <- data.frame()

# Process each fold
cat("\n\nProcessing folds...")
unique_folds <- unique(data$fold_idx)

for (fold in unique_folds) {
  cat("\nProcessing fold", fold, "...")
  
  # Get features for this fold
  fold_features <- top_features %>%
    filter(fold_idx == !!fold) %>%
    pull(feature_name)
  cat("\n  Using", length(fold_features), "features for fold", fold)
  
  # Split data
  train_data <- data %>% filter(fold_idx != !!fold)
  test_data <- data %>% filter(fold_idx == !!fold)
  cat("\n  Training with", nrow(train_data), "samples, testing with", nrow(test_data), "samples")
  
  # Train model and get predictions
  tryCatch({
    result <- train_ridge_model(train_data, test_data, fold_features, "LFC")
    
    # Store predictions
    fold_predictions <- data.frame(
      depmap_id = test_data$depmap_id,
      fold = as.character(fold),  # Convert fold to character
      actual = test_data$LFC,
      predicted = as.vector(result$predictions)
    )
    all_predictions <- bind_rows(all_predictions, fold_predictions)
    
    # Store coefficients
    if (nrow(result$coefficients) > 0) {
      fold_coefficients <- result$coefficients %>%
        mutate(fold = as.character(fold),  # Convert fold to character
               abs_coefficient = abs(coefficient))
      all_coefficients <- bind_rows(all_coefficients, fold_coefficients)
    }
    
    # Calculate metrics
    cor_value <- cor(fold_predictions$actual, fold_predictions$predicted)
    r2_value <- 1 - sum((fold_predictions$actual - fold_predictions$predicted)^2) /
                    sum((fold_predictions$actual - mean(fold_predictions$actual))^2)
    
    fold_metrics <- bind_rows(
      fold_metrics,
      data.frame(
        fold = as.character(fold),  # Convert fold to character
        correlation = cor_value,
        r_squared = r2_value,
        rmse = result$rmse,
        mae = result$mae
      )
    )
    
    cat("\n  Fold", fold, "metrics: Pearson correlation =", round(cor_value, 4),
        ", R² =", round(r2_value, 4))
    
  }, error = function(e) {
    cat("\n  Error in fold", fold, ":", e$message)
  })
}

# Calculate overall performance metrics
cat("\n\nCalculating model performance metrics...")
overall_correlation <- cor(all_predictions$actual, all_predictions$predicted)
overall_r2 <- 1 - sum((all_predictions$actual - all_predictions$predicted)^2) /
                  sum((all_predictions$actual - mean(all_predictions$actual))^2)
overall_rmse <- sqrt(mean((all_predictions$actual - all_predictions$predicted)^2))
overall_mae <- mean(abs(all_predictions$actual - all_predictions$predicted))

# Add overall metrics to fold_metrics
fold_metrics <- bind_rows(
  fold_metrics,
  data.frame(
    fold = "overall",
    correlation = overall_correlation,
    r_squared = overall_r2,
    rmse = overall_rmse,
    mae = overall_mae
  )
)

cat("\nOverall performance:")
cat("\n  Pearson correlation:", round(overall_correlation, 4))
cat("\n  R-squared:", round(overall_r2, 4))
cat("\n  RMSE:", round(overall_rmse, 4))
cat("\n  MAE:", round(overall_mae, 4))

# Summarize feature importance
cat("\n\nSummarizing feature importance...")
feature_importance <- all_coefficients %>%
  group_by(feature) %>%
  summarise(
    mean_importance = mean(abs_coefficient),
    sd_importance = sd(abs_coefficient),
    n_folds = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_importance))

# Save results
cat("\n\nSaving results...")
fwrite(all_predictions, file.path(output_dir, "linear_predictions.csv"))
fwrite(feature_importance, file.path(output_dir, "linear_feature_importance.csv"))
fwrite(fold_metrics, file.path(output_dir, "linear_model_metrics.csv"))

cat("\nLinear model analysis complete.\n") 