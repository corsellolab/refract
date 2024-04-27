import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(input_dim, num_embeddings, embedding_dim, dropout_rate=0.5):
    # Define a simple feedforward neural network structure with an embedding layer, dropout, and batch normalization
    class FeedforwardNeuralNetwork(nn.Module):
        def __init__(self, input_dim, num_embeddings, embedding_dim, dropout_rate):
            super(FeedforwardNeuralNetwork, self).__init__()
            # Define an embedding layer for perturbation names
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            # Define three hidden layers and output layer
            self.fc1 = nn.Linear(input_dim + embedding_dim, 128)  # First hidden layer, adjusted for embedding
            self.fc2 = nn.Linear(128, 64)                        # Second hidden layer
            self.fc3 = nn.Linear(64, 32)                         # Third hidden layer
            self.fc4 = nn.Linear(32, 1)                          # Output layer
            # Dropout layers
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.dropout3 = nn.Dropout(dropout_rate)
            # Batch normalization layers
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(32)

        def forward(self, x, pert_name):
            x = x.float()
            # Embedding for the perturbation name
            pert_embedding = self.embedding(pert_name).float()
            # Concatenate the embedding with the input features
            x = torch.cat([x, pert_embedding], dim=1)
            # Activation functions, batch normalization, and dropout for the hidden layers
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.dropout3(x)
            x = self.fc4(x)  # No activation for output layer, assuming regression task
            return x

    # Create the neural network model
    model = FeedforwardNeuralNetwork(input_dim, num_embeddings, embedding_dim, dropout_rate)
    return model