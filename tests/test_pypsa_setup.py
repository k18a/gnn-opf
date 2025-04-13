import os
import yaml
import pytest
import pypsa
from src.pypsa_setup import load_network_config

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