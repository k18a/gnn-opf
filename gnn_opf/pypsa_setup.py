import pypsa
import yaml

def load_network_config(config_path: str) -> pypsa.Network:
    """
    Load network configuration from YAML file and create a PyPSA network.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        PyPSA Network object configured for DC power flow
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create network with DC settings
    network = pypsa.Network()
    
    # Add carrier for DC power flow
    network.add("Carrier", "DC")
    
    # Add buses
    for bus in config["buses"]:
        network.add("Bus", 
                   name=str(bus["id"]), 
                   v_nom=bus["v_nom"],
                   carrier="DC")
    
    # Add lines with DC parameters
    for line in config["lines"]:
        network.add("Line", 
                   str(line["id"]),
                   bus0=str(line["from_bus"]),
                   bus1=str(line["to_bus"]),
                   x=line["x"],
                   r=1e-4,  # Small non-zero resistance for numerical stability
                   s_nom=line["s_nom"],
                   carrier="DC")
    
    # Add generators
    for gen in config["generators"]:
        network.add("Generator", 
                   str(gen["id"]), 
                   bus=str(gen["bus"]),
                   p_nom=gen["p_nom"], 
                   marginal_cost=gen["marginal_cost"],
                   carrier="DC",
                   p_min_pu=0.0,  # Can go down to 0
                   p_max_pu=1.0)  # Can go up to nominal power
    
    return network 