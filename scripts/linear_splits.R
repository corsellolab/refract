#!/usr/bin/env Rscript

library(tidyverse)
library(data.table)
library(parallel)

cat("Starting linear splits analysis...\n")

# Command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript linear_splits.R <response_file> <feature_file> <output_dir> <seed> <n_cores>")
}

response_file <- args[1]
feature_file <- args[2]
output_dir <- args[3]
seed <- as.numeric(args[4])
n_cores <- as.numeric(args[5])

cat(sprintf("Parameters:\n  Response file: %s\n  Feature file: %s\n  Output directory: %s\n  Random seed: %d\n  Number of cores: %d\n\n", 
            response_file, feature_file, output_dir, seed, n_cores))

# Set random seed
set.seed(seed)

# Create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("Created output directory: %s\n", output_dir))

# Read data
cat("Reading response data...\n")
response_data <- fread(response_file)
cat(sprintf("Found %d rows in response data\n", nrow(response_data)))

cat("Reading feature data...\n")
feature_data <- readRDS(feature_file)
cat(sprintf("Found %d features for %d cell lines\n", ncol(feature_data)-1, nrow(feature_data)))

# Ensure feature_data is in the right format with ModelID column
if (!"ModelID" %in% colnames(feature_data)) {
  stop("Feature file must contain 'ModelID' column")
}

# Ensure response_data has required columns
required_cols <- c("depmap_id", "LFC")
if (!all(required_cols %in% colnames(response_data))) {
  stop("Response file must contain 'depmap_id' and 'LFC' columns")
}

# Join datasets
cat("Joining response and feature data...\n")
joined_data <- response_data %>%
  left_join(feature_data, by = c("depmap_id" = "ModelID"))
cat(sprintf("Joined data has %d rows and %d columns\n", nrow(joined_data), ncol(joined_data)))

# Randomly assign folds
cat("Assigning cell lines to folds...\n")
cell_lines <- unique(joined_data$depmap_id)
n_folds <- 10
fold_assignments <- data.frame(
  depmap_id = cell_lines,
  fold_idx = sample(rep(1:n_folds, length.out = length(cell_lines)))
)
cat(sprintf("Assigned %d cell lines to %d folds\n", length(cell_lines), n_folds))

# Set up parallel processing for folds
cat("\nStarting correlation analysis across all folds using parallel processing...\n")
cl <- makeCluster(n_cores)

# Export necessary objects and functions to the workers
clusterExport(cl, c("joined_data", "fold_assignments"), envir = environment())

# Load required packages and define functions on workers
clusterEvalQ(cl, {
  library(tidyverse)
  library(data.table)
  
  # Function to compute correlation test statistics
  compute_correlation_test <- function(x, y) {
    result <- tryCatch({
      test <- cor.test(x, y, method = "pearson")
      list(
        correlation = test$estimate,
        p_value = test$p.value,
        t_stat = test$statistic,
        df = test$parameter
      )
    }, error = function(e) {
      list(
        correlation = NA,
        p_value = NA,
        t_stat = NA,
        df = NA
      )
    })
    return(result)
  }
  
  # Function to compute correlations for one fold
  compute_fold_correlations <- function(fold_idx, data, fold_assignments) {
    cat(sprintf("Processing fold %d...\n", fold_idx))
    
    # Get training data (all folds except current fold)
    train_data <- data %>%
      left_join(fold_assignments, by = "depmap_id") %>%
      filter(fold_idx != !!fold_idx)
    
    cat(sprintf("  Fold %d: Using %d samples for correlation calculation\n", fold_idx, nrow(train_data)))
    
    # Get feature columns (all except depmap_id, LFC, and fold_idx)
    feature_cols <- setdiff(colnames(train_data), 
                           c("depmap_id", "LFC", "fold_idx", "broad_id", "dose", "name", "ccle_name"))
    
    # Compute correlations for all features
    cat(sprintf("  Fold %d: Computing correlations for %d features\n", fold_idx, length(feature_cols)))
    
    # Calculate correlations and statistics for each feature
    results <- lapply(feature_cols, function(col) {
      complete_cases <- !is.na(train_data[[col]])
      n_complete <- sum(complete_cases)
      
      if (n_complete < 25) {  # Require at least 30 non-NA values for reliable statistics
        return(list(
          correlation = NA,
          p_value = NA,
          t_stat = NA,
          df = NA,
          n_complete = n_complete
        ))
      }
      
      # Compute correlation test statistics
      test_results <- compute_correlation_test(
        train_data[[col]][complete_cases],
        train_data$LFC[complete_cases]
      )
      
      test_results$n_complete <- n_complete
      return(test_results)
    })
    
    # Extract results
    cors <- sapply(results, function(x) x$correlation)
    p_values <- sapply(results, function(x) x$p_value)
    t_stats <- sapply(results, function(x) x$t_stat)
    dfs <- sapply(results, function(x) x$df)
    n_complete_cases <- sapply(results, function(x) x$n_complete)
    
    # Create data frame with results
    result <- data.frame(
      feature_name = feature_cols,
      fold_idx = fold_idx,
      fold_pearson = cors,
      p_value = p_values,
      t_statistic = t_stats,
      degrees_of_freedom = dfs,
      n_complete_cases = n_complete_cases
    ) %>%
      # Remove NA correlations and p-values
      filter(!is.na(p_value)) %>%
      # Rank by p-value (ascending)
      mutate(fold_rank = rank(p_value)) %>%
      # Keep only top 500
      filter(fold_rank <= 500) %>%
      arrange(p_value)
    
    cat(sprintf("  Fold %d: Selected top 500 features from %d valid correlations (min p-value: %.2e)\n", 
                fold_idx, sum(!is.na(p_values)), min(result$p_value)))
    return(result)
  }
})

# Process all folds in parallel
all_correlations <- bind_rows(
  parLapply(cl, 1:n_folds, function(fold) {
    compute_fold_correlations(fold, joined_data, fold_assignments)
  })
)

# Stop the cluster
stopCluster(cl)

# Save results
cat("\nSaving results...\n")
write.csv(fold_assignments, file.path(output_dir, "fold_assignment.csv"), row.names = FALSE)
cat(sprintf("Saved fold assignments to %s\n", file.path(output_dir, "fold_assignment.csv")))

write.csv(all_correlations, file.path(output_dir, "top_features.csv"), row.names = FALSE)
cat(sprintf("Saved feature correlations to %s\n", file.path(output_dir, "top_features.csv")))

cat("\nAnalysis complete!\n")
