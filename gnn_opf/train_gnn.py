import csv
import torch
import torch.nn as nn
import torch.optim as optim
from gnn_opf.pypsa_setup import load_network_config
from gnn_opf.data.graph_data import convert_network_to_graph
from gnn_opf.gnn_opf import PhysicsInformedGNN, physics_penalty
from gnn_opf.data.power_networks import load_network_as_pyg

def read_scenarios(csv_path="data/generated_opf_scenarios.csv"):
    """
    Reads the CSV file containing OPF scenarios.
    Returns a list of dictionaries with keys 'scenario' and 'total_cost'.
    """
    scenarios = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append({
                "scenario": float(row["scenario"]),
                "total_cost": float(row["total_cost"])
            })
    return scenarios

def set_network_loads(network, scenario, load_variation=0.3):
    """
    For each bus in the network, adjust the load based on the scenario number.
    Here, we use a simple dummy mapping: the load is set as:
      load = v_nom * (1 + load_variation * (scenario / 10))
    This can be refined in future iterations.
    """
    for bus in network.buses.index:
        base_load = network.buses.at[bus, "v_nom"]
        variation_factor = scenario / 10.0  # Dummy mapping from scenario number to load variation
        network.buses.at[bus, "load"] = base_load * (1 + load_variation * variation_factor)

def train_gnn(num_epochs=10, learning_rate=0.01):
    """
    Train the PhysicsInformedGNN model using the OPF scenario data.
    For each scenario:
      - Load the network configuration.
      - Update the bus loads using the scenario number.
      - Convert the network to a PyTorch Geometric Data object.
      - Run the model to obtain predictions.
      - Aggregate node predictions by taking their mean (as a proxy for a global metric).
      - Compute the MSE loss between the aggregated prediction and the target total_cost.
      - Add the physics penalty to form the final loss.
      - Backpropagate and update the model.
    Returns the trained model.
    """
    scenarios = read_scenarios()
    model = PhysicsInformedGNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for scenario in scenarios:
            # Load network and update loads for the scenario
            network = load_network_as_pyg('case14')
            set_network_loads(network, scenario["scenario"])
            graph = convert_network_to_graph(network)

            model.train()
            optimizer.zero_grad()
            predictions = model(graph)
            # Aggregate predictions: compute the mean over all nodes
            pred_total = predictions.mean()
            mse_loss = criterion(pred_total, torch.tensor(scenario["total_cost"], dtype=torch.float))
            # Compute a physics penalty (dummy constraint: predictions should be near 100)
            penalty = physics_penalty(graph, predictions)
            loss = mse_loss + penalty
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    return model

if __name__ == "__main__":
    trained_model = train_gnn() 