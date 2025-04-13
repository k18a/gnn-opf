import torch
from gnn_opf.train_gnn import train_gnn
from gnn_opf.gnn_opf import PhysicsInformedGNN, physics_penalty
from gnn_opf.data.power_networks import load_network_as_pyg

def save_model(model, path="model_checkpoint.pth"):
    """
    Save the trained model to the specified path.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path="model_checkpoint.pth", input_dim=1, hidden_dim=16, output_dim=1):
    """
    Load the model from the specified path.
    """
    model = PhysicsInformedGNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def evaluate_model(model, test_csv="data/generated_opf_scenarios.csv"):
    """
    Evaluate the model on the test scenario data.
    This function uses the test scenarios to run inference on the network and calculates a dummy evaluation metric.
    """
    from gnn_opf.train_gnn import read_scenarios, set_network_loads
    from gnn_opf.data.graph_data import convert_network_to_graph
    from gnn_opf.pypsa_setup import load_network_config
    scenarios = read_scenarios(test_csv)
    evaluation_results = []
    for scenario in scenarios:
        # Load network and update loads for each scenario.
        network = load_network_config("config/ieee_14-bus_config.yaml")
        set_network_loads(network, scenario["scenario"])
        graph = convert_network_to_graph(network)
        
        with torch.no_grad():
            predictions = model(graph)
        # Aggregate predictions: mean value over nodes as a proxy metric.
        pred_total = predictions.mean().item()
        target_total = scenario["total_cost"]
        error = abs(pred_total - target_total)
        evaluation_results.append({
            "scenario": scenario["scenario"],
            "predicted_total": pred_total,
            "target_total": target_total,
            "error": error
        })
    return evaluation_results

if __name__ == "__main__":
    # Train the model for demonstration; use a small epoch count if needed.
    model = train_gnn(num_epochs=5, learning_rate=0.01)
    save_model(model, "model_checkpoint.pth")
    loaded_model = load_model("model_checkpoint.pth")
    results = evaluate_model(loaded_model)
    print("Evaluation Results:")
    for res in results:
        print(res)

    # Load IEEE 14-bus network in PyG format
    network = load_network_as_pyg('case14') 