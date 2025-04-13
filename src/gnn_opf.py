import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class PhysicsInformedGNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super(PhysicsInformedGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

def physics_penalty(data, predictions):
    """
    Compute a simple penalty term as a placeholder for physics constraints.
    In this example, we enforce that predictions stay near 100.
    """
    target_value = 100.0  # Dummy physical target for demonstration
    penalty = torch.mean(torch.abs(predictions - target_value))
    return penalty 