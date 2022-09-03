import torch
from torch import nn
from torch.nn.utils import spectral_norm
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        
        x = x + self.fc2(x)
        x = self.relu(self.bn2(x))
        
        x = x + self.fc3(x)
        x = self.relu(self.bn3(x))

        x = x + self.fc4(x)
        x = self.relu(self.bn4(x))
        
        o = self.fc5(x)
        
        return o