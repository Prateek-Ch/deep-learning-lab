import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers, dropout):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if i > 0:
                residual = x  # Store the previous layer's output as residual
            x = conv(x, edge_index)
            if i > 0:
                x = F.relu(x + residual)  # Apply ReLU after adding residual
            else:
                x = F.relu(x)  # Apply ReLU directly if no residual to add
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.sigmoid(x)