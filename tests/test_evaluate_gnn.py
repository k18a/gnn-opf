import os
import torch
from gnn_opf.evaluate_gnn import save_model, load_model, evaluate_model
from gnn_opf.gnn_opf import PhysicsInformedGNN

def test_save_and_load_model(tmp_path):
    # Create a dummy model.
    model = PhysicsInformedGNN()
    # Define a temporary file path.
    temp_file = tmp_path / "temp_model.pth"
    # Save the model.
    save_model(model, str(temp_file))
    # Load the model.
    loaded_model = load_model(str(temp_file))
    # Verify that the loaded model is an instance of PhysicsInformedGNN.
    assert isinstance(loaded_model, PhysicsInformedGNN), "Loaded model is not an instance of PhysicsInformedGNN."

def test_evaluate_model():
    # Train a model for a few epochs.
    from gnn_opf.train_gnn import train_gnn
    model = train_gnn(num_epochs=2, learning_rate=0.01)
    # Evaluate the model using the generated scenario CSV.
    evaluation_results = evaluate_model(model)
    # Check that evaluation results is a non-empty list.
    assert isinstance(evaluation_results, list), "Evaluation results should be a list."
    assert len(evaluation_results) > 0, "Evaluation results list should not be empty."
    # Verify that each result contains required keys.
    for res in evaluation_results:
        for key in ["scenario", "predicted_total", "target_total", "error"]:
            assert key in res, f"Key '{key}' missing in evaluation result." 