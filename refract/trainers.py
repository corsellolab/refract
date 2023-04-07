from sklearn.ensemble import RandomForestRegressor
import logging
import pandas as pd
import os
import numpy as np
import json
from utils import RandomForestConfig

logger = logging.getLogger(__name__)

class RFBaseTrainer:
    """Base trainer for RF model
    """
    def train(self, feature_set, response_set, output_dir, config=RandomForestConfig):
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
            n_jobs=config.n_jobs,   
            random_state=config.random_state
        )

        # get all unique runs
        LFC = response_set.LFC
        runs = LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
        logger.info("Found %d unique runs", len(runs))

        # prepare output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_stats = []
        for feature_name in config.feature_name_list:
            logger.info("Using feature set: %s", feature_name)
            for _, run in runs.iterrows():
                pert_name = run["pert_name"]
                pert_mfc_id = run["pert_mfc_id"]
                dose = run["dose"]
                logger.info("    Training model for run %s, %s, %s", pert_name, pert_mfc_id, dose)

                # get features and response
                X, y = response_set.get_joined_features(
                    pert_name=pert_name,
                    pert_mfc_id=pert_mfc_id,
                    dose=dose,
                    feature_set=feature_set,
                    feature_name=feature_name
                )
                y = y.values.ravel()
                rf.fit(X, y)
                
                # get the feature importances
                importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
                importances = importances.sort_values(by='importance', ascending=False)

                # add data to the imp table
                importances["pert_name"] = pert_name
                importances["pert_mfc_id"] = pert_mfc_id
                importances["dose"] = dose
                # add rank to the improtances
                importances["rank"] = importances["importance"].rank(ascending=False)
                # add the feature name
                importances["feature_name"] = feature_name

                # compute MSE, MSE standard error, R2, Pearson correlation for the  model
                y_pred = rf.predict(X)
                y_true = y
                mse = ((y_true - y_pred) ** 2).mean()
                mse_se = mse / len(y_true)
                r2 = rf.score(X, y)
                pearson = np.corrcoef(y_true, y_pred)[0, 1]
                model_stats.append({
                    "pert_name": pert_name,
                    "pert_mfc_id": pert_mfc_id,
                    "dose": dose,
                    "feature_name": feature_name,
                    "mse": mse,
                    "mse_se": mse_se,
                    "r2": r2,
                    "pearson": pearson
                })

                # save feature importance output
                if not os.path.exists(f"{output_dir}/{pert_name}_{feature_name}"):
                    os.makedirs(f"{output_dir}/{pert_name}_{feature_name}")
                importances.to_csv(f"{output_dir}/{pert_name}_{feature_name}/{pert_mfc_id}_{feature_name}_{dose}.csv")

        # save model stats
        model_stats = pd.DataFrame(model_stats)
        model_stats.to_csv(f"{output_dir}/Model_table.csv")

        # save config
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config.__dict__, f)
