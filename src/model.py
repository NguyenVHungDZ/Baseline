import torch
import torch.nn as nn

class PopulationDensityPredictor(nn.Module):
    
    def __init__(self, num_features: int = 17):
        super().__init__()
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, input_data):
        predicted_density = self.hidden_layers(input_data)
        return predicted_density


def build_model(num_features: int = 17):
    return PopulationDensityPredictor(num_features)

    # k_fold chia 5, theo đúng order training data , rf 500 trees 