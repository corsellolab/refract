import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import shap
import copy

def add_noise(features, noise_level=0.01):
    noise = torch.randn_like(features) * noise_level
    return features + noise

def scale_features(features, scale_factor=0.1):
    scale = 1 + (torch.randn_like(features) * scale_factor)
    return features * scale

class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.fc1(x)))
        out = self.relu2(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X.values, dtype=torch.float32)
        self.targets = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        self.all_features = torch.cat([
            self.features,
            add_noise(self.features),
            scale_features(self.features),
            scale_features(add_noise(self.features))
        ])
        self.all_targets = self.targets.repeat(4, 1)
    
    def __len__(self):
        return len(self.all_features)
    
    def __getitem__(self, idx):
        return self.all_features[idx], self.all_targets[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

    return best_model

def nested_cv(X, y, old_model=None, n_splits_outer=5, n_splits_inner=5, output_size=1, learning_rate=1e-6, batch_size=32, num_epochs=100):
    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)
    input_size = X.shape[1]

    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    all_predictions = np.zeros_like(y, dtype=float)
    all_shap_values = []

    for train_idx, test_idx in kf_outer.split(X):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        best_model = None
        best_val_loss = float('inf')

        kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        for train_idx_inner, val_idx_inner in kf_inner.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer.iloc[train_idx_inner], X_train_outer.iloc[val_idx_inner]
            y_train_inner, y_val_inner = y_train_outer.iloc[train_idx_inner], y_train_outer.iloc[val_idx_inner]
            
            train_dataset = CustomDataset(X_train_inner, y_train_inner)
            val_dataset = CustomDataset(X_val_inner, y_val_inner)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = ImprovedNN(input_size, 512, 256, output_size)
            if old_model is not None:
                model.load_state_dict(old_model)
                for name, param in model.named_parameters():
                    if name not in ['fc3.weight', 'fc3.bias']:
                        param.requires_grad = False
                best_model = old_model

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            
            best_inner_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
            val_loss = sum(criterion(model(features), targets).item() for features, targets in val_loader) / len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = best_inner_model

        # Use the best model to make predictions on the test set
        model.load_state_dict(best_model)
        model.eval()
        
        # Calculate SHAP values using DeepExplainer
        explainer = shap.DeepExplainer(model, torch.tensor(X_train_outer.values, dtype=torch.float))
        shap_values = explainer.shap_values(torch.tensor(X_test_outer.values, dtype=torch.float))
        all_shap_values.append(np.abs(shap_values).mean(axis=0))
        
        with torch.no_grad():
            test_dataset = CustomDataset(X_test_outer, y_test_outer)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_predictions = []
            for features, _ in test_loader:
                outputs = model(features)
                test_predictions.extend(outputs.numpy())
        np.put(all_predictions, test_idx, test_predictions, mode='clip')
        print("Fold complete")

    # Calculate the dataset-wide SHAP values
    average_shap_values = np.mean(all_shap_values, axis=0)
    
    # Create a dictionary of SHAP values with column names as keys
    shap_dict = dict(zip(X.columns, average_shap_values))
    
    # Sort the dictionary by absolute SHAP values
    sorted_shap_dict = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return best_model, all_predictions, sorted_shap_dict

def train(cluster_data_folder, num_epochs=100):
    pretrain_csv = next((file for file in os.listdir(cluster_data_folder) if file.endswith('Pretrain.csv')), None)
    if pretrain_csv is None:
        raise FileNotFoundError("No file ending with 'Pretrain.csv' found in the specified folder.")
    
    cluster_data_path = os.path.join(cluster_data_folder, pretrain_csv)
    drug_files = [os.path.join(cluster_data_folder, file) for file in os.listdir(cluster_data_folder) if file != pretrain_csv]
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(cluster_data_path)
    print("Pretrained Dataframe Loaded")
    X = (df.iloc[:, 2:] - df.iloc[:, 2:].mean()) / df.iloc[:, 2:].std()
    X = X.apply(pd.to_numeric)
    y = df.iloc[:, 1]

    input_size = 501  # Assumes top 500 features and drug id
    output_size = 1
    batch_size = 32
    learning_rate = 0.001

    print("Starting Pretraining")
    stats = {}

    # Pretrain
    pretrained_model, pretrained_preds, pretrained_shap_values = nested_cv(X, y, num_epochs=num_epochs, output_size=output_size)
    pretrained_preds_tensor = torch.tensor(pretrained_preds, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    model_pc = pearsonr(y_tensor.squeeze(), pretrained_preds_tensor.squeeze())
    print(f'Pretraining Model PC: {model_pc}')
    stats['Cluster Level Model'] = model_pc
    cluster_df = pd.DataFrame([pretrained_shap_values])

    models = {}
    num_drugs = len(drug_files)
    for drug_file in drug_files:
        drug_id = os.path.basename(drug_file).split('_')[-1].split('.')[0]
        df = pd.read_csv(drug_file)
        X_finetune = (df.iloc[:, 2:] - df.iloc[:, 2:].mean()) / df.iloc[:, 2:].std()
        y_finetune = df.iloc[:, 1].reset_index(drop=True)
        
        print("Starting finetuning")
        cluster_model, finetune_preds, finetune_shap_values = nested_cv(X_finetune, y_finetune, old_model=pretrained_model, num_epochs=num_epochs, output_size=output_size)
        print("Starting control training")
        raw_model, raw_preds, raw_shap_values = nested_cv(X_finetune, y_finetune, num_epochs= (num_drugs+1) * num_epochs, output_size=output_size)
        
        y_finetune_tensor = torch.tensor(y_finetune.values, dtype=torch.float32).unsqueeze(1)
        finetune_preds_tensor = torch.tensor(finetune_preds, dtype=torch.float32)
        raw_preds_tensor = torch.tensor(raw_preds, dtype=torch.float32)
        finetune_pc = pearsonr(y_finetune_tensor.squeeze(), finetune_preds_tensor.squeeze())
        raw_pc = pearsonr(y_finetune_tensor.squeeze(), raw_preds_tensor.squeeze())

        stats[f'Finetuned_Model_{drug_id}'] = finetune_pc
        stats[f'Raw_Model_{drug_id}'] = raw_pc

        finetune_df = pd.DataFrame([finetune_shap_values])
        raw_df = pd.DataFrame([raw_shap_values])

        models[drug_id] = (finetune_df, raw_df)
        print(f"Drug: {drug_id}, Finetuned PC: {finetune_pc}, Standard PC: {raw_pc}")

    return pretrained_model, cluster_df, stats, models

def main():
    return

if __name__ == "__main__":
    main()
