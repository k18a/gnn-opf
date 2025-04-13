"""
Power network data handling utilities.
This module provides functions to load various power system test cases and convert them
between different graph formats (pandapower, networkx, pytorch geometric).
"""

import pandapower.networks as pn
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def get_case_function_map():
    """Returns a dictionary mapping case names to their loading functions."""
    return {
        'case4gs': pn.case4gs,
        'case5': pn.case5,
        'case6ww': pn.case6ww,
        'case9': pn.case9,
        'case14': pn.case14,
        'case24_ieee_rts': pn.case24_ieee_rts,
        'case30': pn.case30,
        'case_ieee30': pn.case_ieee30,
        'case33bw': pn.case33bw,
        'case39': pn.case39,
        'case57': pn.case57,
        'case89pegase': pn.case89pegase,
        'case118': pn.case118,
        'case145': pn.case145,
        'case300': pn.case300,
        'case1354pegase': pn.case1354pegase,
        'case1888rte': pn.case1888rte,
        'case2848rte': pn.case2848rte,
        'case2869pegase': pn.case2869pegase,
        'case3120sp': pn.case3120sp,
        'case6470rte': pn.case6470rte,
        'case6495rte': pn.case6495rte,
        'case6515rte': pn.case6515rte,
        'case9241pegase': pn.case9241pegase,
        'GBnetwork': pn.GBnetwork,
        'GBreducednetwork': pn.GBreducednetwork,
        'iceland': pn.iceland
    }

def load_power_network(case_name):
    """
    Load a power network test case by name.
    
    Args:
        case_name (str): Name of the test case to load
        
    Returns:
        pandapower.auxiliary.pandapowerNet: The loaded network
        
    Raises:
        ValueError: If case_name is not recognized
    """
    case_functions = get_case_function_map()
    if case_name not in case_functions:
        raise ValueError(f"Unknown case name: {case_name}. Available cases: {list(case_functions.keys())}")
    
    return case_functions[case_name]()

def build_graph_from_pandapower(net):
    """
    Convert a pandapower network to a networkx graph.
    
    Args:
        net (pandapower.auxiliary.pandapowerNet): The pandapower network
        
    Returns:
        networkx.Graph: The converted graph with voltage and line parameters as attributes
    """
    G = nx.Graph()

    # Add nodes (buses)
    for idx, row in net.bus.iterrows():
        G.add_node(idx, voltage=row.vn_kv)

    # Add edges (lines)
    for idx, row in net.line.iterrows():
        G.add_edge(
            row.from_bus, 
            row.to_bus, 
            r_ohm=row.r_ohm_per_km * row.length_km,
            x_ohm=row.x_ohm_per_km * row.length_km, 
            capacity=row.max_i_ka
        )

    return G

def to_pyg_data(nx_graph):
    """
    Convert a networkx graph to a PyTorch Geometric Data object.
    
    Args:
        nx_graph (networkx.Graph): The input networkx graph
        
    Returns:
        torch_geometric.data.Data: The converted PyG Data object
    """
    return from_networkx(nx_graph)

def load_network_as_pyg(case_name):
    """
    Convenience function to load a power network and convert it directly to PyG format.
    
    Args:
        case_name (str): Name of the test case to load
        
    Returns:
        torch_geometric.data.Data: The network as a PyG Data object
    """
    pp_net = load_power_network(case_name)
    nx_graph = build_graph_from_pandapower(pp_net)
    return to_pyg_data(nx_graph) 