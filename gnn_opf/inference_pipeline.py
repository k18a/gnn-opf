import torch
from gnn_opf.pypsa_setup import load_network_config
from gnn_opf.data.graph_data import convert_network_to_graph
from gnn_opf.gnn_opf import PhysicsInformedGNN, physics_penalty
from gnn_opf.data.power_networks import load_network_as_pyg

def run_inference():
    # Load IEEE 14-bus network in PyG format
    network = load_network_as_pyg('case14')
    
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