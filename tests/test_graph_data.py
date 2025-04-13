import os
import pytest
import torch
from gnn_opf.pypsa_setup import load_network_config
from gnn_opf.data.graph_data import convert_network_to_graph

def test_convert_network_to_graph():
    # Load the PyPSA network using the existing configuration file.
    network = load_network_config("config/ieee_14-bus_config.yaml")
    # Convert the network to a PyTorch Geometric Data object.
    data = convert_network_to_graph(network)
    
    # Test: The number of nodes should equal the number of buses.
    assert data.x.size(0) == len(network.buses), "Number of nodes does not match number of buses."
    
    # Test: The edge_index tensor must have shape [2, num_edges].
    assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2, "edge_index tensor shape is incorrect."
    
    # Test: Each edge must have 2 attributes (reactance and capacity).
    assert data.edge_attr.size(1) == 2, "Each edge attribute should have 2 elements." 