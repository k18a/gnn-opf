import os
import yaml
import pytest
import pypsa
from gnn_opf.pypsa_setup import load_network_config
from gnn_opf.data.power_networks import load_power_network, load_network_as_pyg
import pandapower
from torch_geometric.data import Data

def test_config_file_exists():
    config_path = "config/ieee_14-bus_config.yaml"
    assert os.path.exists(config_path), "The config file does not exist."
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    for key in ["buses", "lines", "generators"]:
        assert key in config, f"Key '{key}' not found in configuration file."

def test_load_network_config():
    network = load_network_config("config/ieee_14-bus_config.yaml")
    assert isinstance(network, pypsa.Network), "load_network_config did not return a PyPSA network."
    # For the IEEE 14-bus case, assume there should be exactly 14 buses.
    assert len(network.buses) == 14, "Number of buses does not match the config."
    # Also check that at least one line and one generator exists.
    assert len(network.lines) >= 1, "Expected at least one line in the network."
    assert len(network.generators) >= 1, "Expected at least one generator in the network."

def test_load_power_network():
    """Test loading a power network test case."""
    network = load_power_network('case14')
    assert isinstance(network, pandapower.auxiliary.pandapowerNet), "load_power_network did not return a pandapower network."

def test_load_network_as_pyg():
    """Test loading a power network directly as PyG data."""
    data = load_network_as_pyg('case14')
    assert isinstance(data, Data), "load_network_as_pyg did not return a PyG Data object."

def test_invalid_case_name():
    """Test that loading an invalid case name raises an error."""
    with pytest.raises(ValueError):
        load_power_network('nonexistent_case') 