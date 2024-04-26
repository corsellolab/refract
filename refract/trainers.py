# trainer for the XGBoost ranking model
import logging
from functools import partial

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from flaml import AutoML
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

def get_n_correlated_features(y, X, colnames, n):
    # Step 1: Compute the correlation for each column
    correlations = np.array([np.corrcoef(y, X[:, i])[0, 1] for i in range(X.shape[1])])

    # Step 2: Get the top n correlated features
    top_n_indices = np.argsort(np.abs(correlations))[-n:]
    top_n_colnames = [colnames[i] for i in top_n_indices]

    # Step 3: Return the list of column names
    return top_n_colnames

def get_correlated_features(y, X, colnames, p):
    # Step 1: Compute the correlation for each column
    correlations = np.array([np.corrcoef(y, X[:, i])[0, 1] for i in range(X.shape[1])])
    
    # Step 2: Group columns by TYPE
    type_dict = {}
    for i, colname in enumerate(colnames):
        type_name = colname.split("_")[0]
        if type_name not in type_dict:
            type_dict[type_name] = []
        type_dict[type_name].append((correlations[i], colname))
    
    # Step 3: Sample the top p proportion of correlated features within each type
    selected_colnames = []
    for type_name in type_dict:
        if type_name != "LIN":
            sorted_correlations = sorted(type_dict[type_name], key=lambda x: -abs(x[0]))  # sort by absolute correlation value in descending order
            top_p_count = int(p * len(sorted_correlations))
            top_p_colnames = [colname for _, colname in sorted_correlations[:top_p_count]]
            selected_colnames.extend(top_p_colnames)
        else:
            sorted_correlations = sorted(type_dict[type_name], key=lambda x: -abs(x[0]))  # sort by absolute correlation value in descending order
            top_p_count = int(p * len(sorted_correlations))
            top_p_colnames = [colname for _, colname in sorted_correlations]
            selected_colnames.extend(top_p_colnames)

    
    # Step 4: Return the list of column names
    return selected_colnames

class BaselineTrainer:
    def __init__(
        self,
        response_train,
        response_test,
        feature_df,
        response_col="LFC.cb",
        cell_line_col="ccle_name",
        num_features=500,
    ):
        self.response_train = response_train
        self.response_test = response_test
        self.feature_df = feature_df
        self.response_col = response_col
        self.cell_line_col = cell_line_col
        self.num_features = num_features

        self.top_feature_names = None
        self.model = None

        self.X_test_df = None
        self.y_test = None
        self.cell_line_test = None
        self.y_test_pred = None
        self.shap_df = None
        self.test_corr = None


    def train(self):
        # select appropriate cell lines
        X_train_df = self.feature_df.loc[self.response_train[self.cell_line_col], :]
        X_test_df = self.feature_df.loc[self.response_test[self.cell_line_col], :]
        
        # drop all columns with zero stddev
        X_train_df = X_train_df.loc[:, X_train_df.std() != 0]

        # get X_train, X_test y_train, y_test
        X_train = X_train_df.values
        X_test = X_test_df.values
        y_train = self.response_train[self.response_col].values
        y_test = self.response_test[self.response_col].values

        # filter to top features
        top_features = get_n_correlated_features(y_train, X_train, X_train_df.columns, n=500)

        # filter features
        X_train_df = X_train_df.loc[:, top_features]
        X_test_df = X_test_df.loc[:, top_features]
        X_train = X_train_df.values
        X_test = X_test_df.values

        # train a random forest model
        model = RandomForestRegressor(
            n_estimators=500,        # num.trees in ranger
            criterion='squared_error',  # variance in ranger for regression
            max_features=.33,     # mtry, 'auto' sets to n_features / 3 for regression
            min_samples_split=2,     # min.node.size, more a function of min impurity decrease
            min_samples_leaf=5,      # min.node.size in ranger for regression
            bootstrap=True,          # replace in ranger
            oob_score=True,          # oob.error in ranger
            n_jobs=-1,               # Use all cores, similar to default parallelism in ranger
            random_state=None,       # Control randomness for reproducibility
            verbose=0                # verbose in ranger
        )

        model.fit(X_train, y_train)
        self.top_feature_names = X_train_df.columns
        self.model = model
        
        # predict
        y_test_pred = model.predict(X_test)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=X_test_df.columns)

        # print fold correlation
        print(f"Fold correlation: {pearsonr(y_test, y_test_pred)[0]}")

        # save to self
        self.X_test_df = X_test_df
        self.y_test = y_test
        self.cell_line_test = self.response_test[self.cell_line_col]
        self.y_test_pred = y_test_pred
        self.shap_df = shap_values_df
        self.test_corr = pearsonr(y_test, y_test_pred)[0]


class AutoMLTrainer:
    """Trains a LGBM Regression Model"""

    def __init__(
        self,
        response_train,
        response_test,
        feature_df,
        response_col="LFC.cb",
        cell_line_col="ccle_name",
        num_features=500,
    ):
        self.response_train = response_train
        self.response_test = response_test
        self.feature_df = feature_df
        self.response_col = response_col
        self.cell_line_col = cell_line_col
        self.num_features = num_features

        self.top_feature_names = None
        self.model = None

        self.X_test_df = None
        self.y_test = None
        self.cell_line_test = None
        self.y_test_pred = None
        self.shap_df = None
        self.test_corr = None


    def train(self):
        X_train_df = self.feature_df.loc[self.response_train[self.cell_line_col], :]
        X_test_df = self.feature_df.loc[self.response_test[self.cell_line_col], :]
        
        # drop all columns with zero stddev
        X_train_df = X_train_df.loc[:, X_train_df.std() != 0]

        # get X_train, X_test y_train, y_test
        X_train = X_train_df.values
        X_test = X_test_df.values
        y_train = self.response_train[self.response_col].values
        y_test = self.response_test[self.response_col].values

        # filter to top features
        top_features = get_n_correlated_features(y_train, X_train, X_train_df.columns, n=500)

        # filter features
        X_train_df = X_train_df.loc[:, top_features]
        X_test_df = X_test_df.loc[:, top_features]
        X_train = X_train_df.values
        X_test = X_test_df.values

        automl = AutoML()
        automl.fit(
            X_train, 
            y_train, 
            task="regression", 
            time_budget=120, 
            metric="rmse", 
            estimator_list=['xgboost', 'rf', 'lgbm'],
        )
        self.top_feature_names = X_train_df.columns
        self.model = automl.model.estimator
        self.automl = automl
        
        # predict
        y_test_pred = automl.predict(X_test)
        explainer = shap.TreeExplainer(automl.model.estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=X_test_df.columns)

        # print fold correlation
        print(f"Fold correlation: {pearsonr(y_test, y_test_pred)[0]}")

        # save to self
        self.X_test_df = X_test_df
        self.y_test = y_test
        self.cell_line_test = self.response_test[self.cell_line_col]
        self.y_test_pred = y_test_pred
        self.shap_df = shap_values_df
        self.test_corr = pearsonr(y_test, y_test_pred)[0]



def get_model(input_dim):
    # Define a simple feedforward neural network structure
    class FeedforwardNeuralNetwork(nn.Module):
        def __init__(self, input_dim):
            super(FeedforwardNeuralNetwork, self).__init__()
            # Define three hidden layers and output layer
            self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
            self.dropout1 = nn.Dropout(0.5)       # Dropout layer after first hidden layer
            self.fc2 = nn.Linear(128, 64)         # Second hidden layer
            self.dropout2 = nn.Dropout(0.5)       # Dropout layer after second hidden layer
            self.fc3 = nn.Linear(64, 32)          # Third hidden layer
            self.dropout3 = nn.Dropout(0.5)       # Dropout layer after third hidden layer
            self.fc4 = nn.Linear(32, 1)           # Output layer

        def forward(self, x):
            x = F.relu(self.fc1(x))               # Activation function for first layer
            x = self.dropout1(x)                  # Apply dropout after first layer
            x = F.relu(self.fc2(x))               # Activation function for second layer
            x = self.dropout2(x)                  # Apply dropout after second layer
            x = F.relu(self.fc3(x))               # Activation function for third layer
            x = self.dropout3(x)                  # Apply dropout after third layer
            x = self.fc4(x)                       # No activation for output layer, assuming regression task
            return x

    # Create the neural network model
    model = FeedforwardNeuralNetwork(input_dim)
    return model


def train_model(model, train_dataloader, val_dataloader, num_epochs, patience):
    # Assuming the loss function and optimizer are predefined globally or are parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataloader.dataset)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataloader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load the best model back
    model.load_state_dict(torch.load('best_model.pth'))

class NNTrainer:
    def __init__(
        self,
        response_train,
        response_test,
        feature_df,
        response_col="LFC.cb",
        cell_line_col="ccle_name",
        num_features=500,
    ):
        self.response_train = response_train
        self.response_test = response_test
        self.feature_df = feature_df
        self.response_col = response_col
        self.cell_line_col = cell_line_col
        self.num_features = num_features

        self.top_feature_names = None
        self.model = None

        self.X_test_df = None
        self.y_test = None
        self.cell_line_test = None
        self.y_test_pred = None
        self.shap_df = None
        self.test_corr = None


    def train(self):
        X_train_df = self.feature_df.loc[self.response_train[self.cell_line_col], :]
        X_test_df = self.feature_df.loc[self.response_test[self.cell_line_col], :]

        # drop all columns with zero stddev
        X_train_df = X_train_df.loc[:, X_train_df.std() != 0]

        # get X_train, X_test y_train, y_test
        X_train = X_train_df.values
        X_test = X_test_df.values
        y_train = self.response_train[self.response_col].values
        y_test = self.response_test[self.response_col].values

        # get top features
        top_features = get_n_correlated_features(y_train, X_train, X_train_df.columns, n=500)

        # filter features
        X_train_df = X_train_df.loc[:, top_features]
        X_test_df = X_test_df.loc[:, top_features]
        X_train = X_train_df.values
        X_test = X_test_df.values

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # to tensor
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        # squeeze y
        y_train = y_train.squeeze()
        y_val = y_val.squeeze()
        y_test = y_test.squeeze()

        # get a dataset object
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # get dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train the model
        model = get_model(X_train.shape[1])
        train_model(model, train_loader, val_loader, num_epochs=100, patience=5)

        self.top_feature_names = X_train_df.columns
        self.model = model
        
        # predict
        y_test_pred = model(torch.FloatTensor(X_test)).detach().numpy().flatten()
        # get shap values
        explainer = shap.DeepExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test.squeeze(), check_additivity=False)
        shap_values_df = pd.DataFrame(np.squeeze(shap_values), columns=X_test_df.columns)

        # print fold correlation
        print(f"Fold correlation: {pearsonr(y_test, y_test_pred)[0]}")

        # save to self
        self.X_test_df = X_test_df
        self.y_test = y_test
        self.cell_line_test = self.response_test[self.cell_line_col]
        self.y_test_pred = y_test_pred
        self.shap_df = shap_values_df
        self.test_corr = pearsonr(y_test, y_test_pred)[0]