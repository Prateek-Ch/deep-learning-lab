import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers, dropout):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            if i > 0:
                x = x + conv(x, edge_index)  # Residual connection
            else:
                x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        
        return F.sigmoid(x)
