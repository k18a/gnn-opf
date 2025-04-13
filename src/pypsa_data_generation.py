import random
import csv
import pypsa
from src.pypsa_setup import load_network_config

def generate_opf_scenarios(num_scenarios: int = 10, load_variation: float = 0.3) -> str:
    random.seed(42)  # Ensure reproducibility
    network = load_network_config("config/ieee_14-bus_config.yaml")
    
    # Here, we assume that each bus's load is initially set to its nominal voltage value.
    # This is a placeholder; adjust if your network has an explicit 'load' attribute.
    base_loads = {bus: network.buses.at[bus, "v_nom"] for bus in network.buses.index}
    
    results = []
    
    for i in range(num_scenarios):
        # For each scenario, vary the load on each bus randomly by ±load_variation (e.g., 30%)
        for bus in network.buses.index:
            variation = (random.random() * 2 - 1) * load_variation  # value in [-load_variation, load_variation]
            network.buses.at[bus, "load"] = base_loads[bus] * (1 + variation)
        
        # Solve the DC OPF for the current scenario.
        # Here we use the DC power flow solver; if you have a specific OPF solver, replace this.
        network.lpf()
        
        # Collect key results from the scenario.
        scenario_result = {
            "scenario": i + 1,
            "total_cost": sum(network.generators.marginal_cost * network.generators_t.p.iloc[0]),
            # You can also add additional keys (e.g., bus angles, line flows) if required.
        }
        results.append(scenario_result)
    
    # Define output file path.
    output_file = "data/generated_opf_scenarios.csv"
    
    # Write the results to a CSV file.
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    
    return output_file 