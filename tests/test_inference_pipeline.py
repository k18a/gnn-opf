import torch
from gnn_opf.inference_pipeline import run_inference

def test_run_inference():
    predictions, penalty = run_inference()
    # Check that predictions is a tensor with at least one node output.
    assert isinstance(predictions, torch.Tensor), "Predictions should be a tensor."
    assert predictions.dim() == 2 and predictions.size(1) == 1, "Predictions should have shape [num_nodes, 1]."
    # Check that the penalty is a non-negative float.
    assert isinstance(penalty, float), "Penalty should be a float."
    assert penalty >= 0, "Penalty should be non-negative." 