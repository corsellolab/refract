library(plyr)
library(tidyverse)
library(magrittr)
library(ranger)
library(reshape2)
library(lmtest)
library(limma)
library(corpcor)
library(ashr)

random_forest <- function(X_train, X_test, Y_train, Y_test, n = 500) {
    X_train <- as.matrix(X_train)
    X_test <- as.matrix(X_test)
    
    # Remove columns with all NAs or zero variance
    valid_cols <- !is.na(colnames(X_train)) & apply(X_train, 2, var, na.rm = TRUE) > 0
    X_train <- X_train[, valid_cols, drop = FALSE]
    X_test <- X_test[, valid_cols, drop = FALSE]
    
    # Sanitize column names
    colnames(X_train) <- make.names(colnames(X_train), unique = TRUE)
    colnames(X_test) <- make.names(colnames(X_test), unique = TRUE)

    y <- Y_train$LFC
    
    # Replace NAs with column medians
    replace_na_with_median <- function(x) {
        if (all(is.na(x))) return(rep(0, length(x)))  # Handle fully NA columns
        x[is.na(x)] <- median(x, na.rm = TRUE)
        return(x)
    }
    
    X_train <- apply(X_train, 2, replace_na_with_median)
    X_test <- apply(X_test, 2, replace_na_with_median)
    
    # Convert back to data frame after imputation
    X_train <- as.data.frame(X_train)
    X_test <- as.data.frame(X_test)

    # Feature selection: Select top correlated features
    correlations <- apply(X_train, 2, function(x) {
        if (all(is.na(x)) || var(x, na.rm = TRUE) == 0) return(NA)  # Avoid errors
        cor(x, y, use="complete.obs")
    })
    
    abs_correlations <- abs(correlations)
    abs_correlations[is.na(abs_correlations)] <- 0  # Replace NAs with zero
    top_features <- names(sort(abs_correlations, decreasing=TRUE)[seq_len(min(n, length(abs_correlations)))])
    
    X_train <- X_train[, top_features, drop = FALSE]
    X_test <- X_test[, top_features, drop = FALSE]
    
    train_df <- as.data.frame(cbind(X_train, y_label = y))

    # Train the random forest model
    rf <- ranger(y_label ~ ., data = train_df, importance = "impurity")

    # Predict on test set
    test_df <- as.data.frame(X_test)
    predictions <- predict(rf, data = test_df)$predictions

    # Extract feature importance
    ss <- tibble(feature = names(rf$variable.importance),
                 RF.imp = rf$variable.importance / sum(rf$variable.importance))

    return(list(rf.fit = ss, preds = predictions, true = Y_test$LFC))
}

#feature_path <- "/Users/nick/Desktop/new_refract/data/features/processed_features/x-all.rds"
#response_file <- "/Users/nick/Desktop/new_refract/data/responses/lowercase_responses/amg-232_2.5.csv"
#output_dir <- "test_baseline_dir"
#fold_dir <- "/Users/nick/Desktop/new_refract/data/test_output/train_splits/amg-232_2.5"

# make above variables set from command line
args <- commandArgs(trailingOnly = TRUE)
feature_path <- args[1]
response_file <- args[2]
output_dir <- args[3]
fold_dir <- args[4]

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

Xall <- readRDS(feature_path) %>%
  column_to_rownames(var = "ModelID")

Yall <- read.csv(response_file) %>%
  column_to_rownames(var = "depmap_id")
Yall$LFC <- as.numeric(Yall$LFC)

fold_files <- list.files(fold_dir, pattern = "\\.split.txt$", full.names = TRUE)
fold_assignment <- bind_rows(lapply(fold_files, function(file) {
  fold_idx <- as.numeric(gsub("\\.split.txt", "", basename(file)))
  data.table::fread(file) %>% mutate(fold_idx = fold_idx)
}))

intersecting_rownames <- intersect(rownames(Xall), rownames(Yall))
Xall <- Xall[intersecting_rownames, , drop = FALSE]
Yall <- Yall[intersecting_rownames, , drop = FALSE]

unique_folds <- unique(fold_assignment$fold_idx)
SS <- tibble()
preds <- tibble()
for (fold in unique_folds) {
  print(paste("Running fold", fold))
  this_fold <- fold_assignment %>% filter(fold_idx == fold)
  train_depmap_ids <- this_fold %>% filter(split == "train") %>% pull(depmap_id)
  test_depmap_ids <- this_fold %>% filter(split == "test") %>% pull(depmap_id)
  
  Xtrain <- Xall[train_depmap_ids, , drop = FALSE]
  Xtest <- Xall[test_depmap_ids, , drop = FALSE]
  Ytrain <- Yall[train_depmap_ids, , drop = FALSE]
  Ytest <- Yall[test_depmap_ids, , drop = FALSE]
  
  fold_res <- random_forest(Xtrain, Xtest, Ytrain, Ytest, n=500)
  rf.fit <- fold_res$rf.fit
  rf.fit$fold <- fold
  pred_true <- tibble(
    ModelID = test_depmap_ids,  # Add ModelID column
    pred = fold_res$preds, 
    true = fold_res$true
  )
  pred_true$fold <- fold
  preds <- bind_rows(preds, pred_true)
  SS <- bind_rows(SS, rf.fit)
}

RF.importances <- reshape2::acast(SS, feature ~ fold, value.var = "RF.imp")

RF.table <- tibble(feature = rownames(RF.importances),
                   RF.imp.mean = apply(RF.importances, 1, mean, na.rm = TRUE),
                   RF.imp.sd = apply(RF.importances, 1, sd, na.rm = TRUE),
                   RF.imp.stability = apply(RF.importances, 1, function(x) mean(!is.na(x)))) %>%
  filter(RF.imp.stability > 0.5, feature != "(Intercept)") %>%
  arrange(desc(RF.imp.mean))  # Sort by descending mean importance

mse <- mean((preds$pred - preds$true)^2, na.rm = TRUE)
mse.se <- sqrt(var((preds$pred - preds$true)^2, na.rm = TRUE))/sqrt(nrow(preds))
r2 <- 1 - (mse / var(preds$true, na.rm = TRUE))
ps <- cor(preds$true, preds$pred, use = "pairwise.complete.obs")

model_table <- tibble(MSE = mse, MSE.se = mse.se, R2 = r2, PearsonScore = ps)

write.csv(model_table, file.path(output_dir, "Model_table.csv"), row.names = FALSE)
write.csv(RF.table, file.path(output_dir, "RF_table.csv"), row.names = FALSE)
write.csv(preds, file.path(output_dir, "pred_true.csv"), row.names = FALSE)