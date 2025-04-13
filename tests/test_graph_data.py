import pytest
from gnn_opf.data.power_networks import load_power_network, build_graph_from_pandapower, to_pyg_data
import networkx as nx
from torch_geometric.data import Data

def test_graph_conversion():
    """Test the complete graph conversion pipeline."""
    # Load a test network
    pp_net = load_power_network('case14')
    
    # Convert to networkx
    nx_graph = build_graph_from_pandapower(pp_net)
    assert isinstance(nx_graph, nx.Graph), "Graph conversion failed"
    assert len(nx_graph.nodes) > 0, "Graph has no nodes"
    assert len(nx_graph.edges) > 0, "Graph has no edges"
    
    # Convert to PyG
    pyg_data = to_pyg_data(nx_graph)
    assert isinstance(pyg_data, Data), "PyG conversion failed"
    assert pyg_data.num_nodes == len(nx_graph.nodes), "Node count mismatch"
    assert pyg_data.num_edges == len(nx_graph.edges) * 2, "Edge count mismatch (PyG uses directed edges)" 