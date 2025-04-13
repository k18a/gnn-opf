import pypsa
import yaml

def load_network_config(config_path: str) -> pypsa.Network:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    network = pypsa.Network()
    for bus in config["buses"]:
        network.add("Bus", name=str(bus["id"]), v_nom=bus["voltage"])
    for line in config["lines"]:
        network.add("Line", str(line["id"]),
                    bus0=str(line["from_bus"]),
                    bus1=str(line["to_bus"]),
                    x=line["reactance"],
                    s_nom=line["capacity"])
    for gen in config["generators"]:
        network.add("Generator", str(gen["id"]), bus=str(gen["bus"]),
                    p_nom=gen["max_output"], marginal_cost=gen["cost"])
    return network 