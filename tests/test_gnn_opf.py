import torch
from torch_geometric.data import Data
import pytest
from gnn_opf.gnn_opf import PhysicsInformedGNN, physics_penalty

def test_gnn_forward():
    # Create a dummy graph Data object with 4 nodes and a simple edge index.
    x = torch.tensor([[230.0], [230.0], [230.0], [230.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 3, 0],
                               [1, 0, 3, 2, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    model = PhysicsInformedGNN()
    output = model(data)
    # Check that output has the same number of nodes and one feature per node.
    assert output.shape == (4, 1), "Output shape should be [number_of_nodes, 1]."

def test_physics_penalty():
    # Create dummy predictions and compute penalty.
    predictions = torch.tensor([[90.0], [110.0], [100.0]], dtype=torch.float)
    penalty = physics_penalty(None, predictions)
    # Penalty should be a positive scalar.
    assert torch.is_tensor(penalty) and penalty.dim() == 0, "Penalty should be a scalar tensor."
    assert penalty.item() >= 0, "Penalty should be non-negative." 