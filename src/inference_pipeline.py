import torch
from src.pypsa_setup import load_network_config
from src.data.graph_data import convert_network_to_graph
from src.gnn_opf import PhysicsInformedGNN, physics_penalty

def run_inference():
    # Load the network from the configuration file.
    network = load_network_config("config/ieee_14-bus_config.yaml")
    
    # Convert the network to a PyTorch Geometric Data object.
    graph_data = convert_network_to_graph(network)
    
    # Initialize the PhysicsInformedGNN model.
    model = PhysicsInformedGNN()
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Run the forward pass to get predictions.
    with torch.no_grad():
        predictions = model(graph_data)
    
    # Compute the physics penalty as a simple check.
    penalty = physics_penalty(graph_data, predictions)
    
    # Print the predicted outputs and the physics penalty.
    print("Predictions:")
    print(predictions)
    print("Physics Penalty:")
    print(penalty.item())
    
    return predictions, penalty.item()

if __name__ == "__main__":
    run_inference() 