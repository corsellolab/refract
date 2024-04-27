import torch
from sklearn.preprocessing import LabelEncoder

class PrismDataset(torch.utils.data.Dataset):
    def __init__(self, feature_df, response_df, drug_name_encoder, \
            cell_line_col="ccle_name", response_col="LFC.cb", drug_col="pert_name"):
        # save input
        self.feature_df = feature_df
        self.response_df = response_df
        self.drug_name_encoder = drug_name_encoder
        self.cell_line_col = cell_line_col
        self.response_col = response_col
        self.drug_col = drug_col

        # encode drug names
        self.drug_names = self.response_df[self.drug_col].values
        self.drug_names = self.drug_name_encoder.transform(self.drug_names)

        # utility to increase ~speed~
        self.num_embeddings = len(self.drug_name_encoder.classes_)
        self.num_features = self.feature_df.shape[1]
        self.responses = self.response_df[self.response_col].values
        self.cell_line = self.response_df[self.cell_line_col].values

        # convert everything to tensors
        self.responses = torch.tensor(self.responses, dtype=torch.float32)
        self.drug_names = torch.tensor(self.drug_names, dtype=torch.long)
        self.feature_dict = {}
        for index, row in self.feature_df.iterrows():
            self.feature_dict[index] = torch.tensor(row.values, dtype=torch.float32)

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        response = self.responses[idx]
        drug = self.drug_names[idx]
        cell_line = self.cell_line[idx]
        features = self.feature_dict[cell_line]

        return features, drug, response