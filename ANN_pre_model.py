
import torch
import torch.nn as nn
import torch.optim as optim


class ANN_2(nn.Module):
    def __init__(self):
        super(ANN_2, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer with 3 neurons for Precipitation, HQprecipitation, IRprecipitation
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x