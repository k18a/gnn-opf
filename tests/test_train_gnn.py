import os
import torch
import pytest
from gnn_opf.train_gnn import read_scenarios, set_network_loads, train_gnn
from gnn_opf.pypsa_setup import load_network_config

def test_read_scenarios():
    csv_path = "data/generated_opf_scenarios.csv"
    assert os.path.exists(csv_path), "CSV file for scenarios does not exist."
    scenarios = read_scenarios(csv_path)
    assert isinstance(scenarios, list), "Scenarios should be returned as a list."
    assert len(scenarios) > 0, "Scenarios list should not be empty."
    for scenario in scenarios:
        assert "scenario" in scenario and "total_cost" in scenario, "Each scenario must have 'scenario' and 'total_cost' keys."

def test_set_network_loads():
    network = load_network_config("config/ieee_14-bus_config.yaml")
    original_loads = {bus: network.buses.at[bus, "v_nom"] for bus in network.buses.index}
    # Set loads for a dummy scenario value, e.g., scenario = 5
    set_network_loads(network, scenario=5, load_variation=0.3)
    for bus in network.buses.index:
        new_load = network.buses.at[bus, "load"]
        assert new_load != original_loads[bus], "Bus load should be updated from its nominal value."

def test_training_pipeline():
    # Run training for a small number of epochs and check the trained model type.
    model = train_gnn(num_epochs=2, learning_rate=0.01)
    assert isinstance(model, torch.nn.Module), "Trained model should be an instance of torch.nn.Module." 