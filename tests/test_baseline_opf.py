import os
import torch
from src.baseline_opf import OPFDataset, BaselineOPFModel, train_baseline_model

def test_dataset_loading():
    csv_file = "data/generated_opf_scenarios.csv"
    # Check that the CSV file exists.
    assert os.path.exists(csv_file), "Generated CSV file does not exist."
    dataset = OPFDataset(csv_file)
    # Check that the dataset contains at least one sample.
    assert len(dataset) > 0, "Dataset is empty."
    # Check that a sample returns a feature tensor of shape [1] and target tensor of shape [1]
    feature, target = dataset[0]
    assert feature.shape[0] == 1, "Feature tensor should have dimension [1]."
    assert target.shape[0] == 1, "Target tensor should have dimension [1]."

def test_model_forward():
    # Create a sample model and test forward pass.
    model = BaselineOPFModel()
    test_input = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    output = model(test_input)
    # Output should have shape [3, 1]
    assert output.shape == (3, 1), "Model output shape is incorrect."

def test_training():
    # Run training for a few epochs and check that the model parameters change.
    csv_file = "data/generated_opf_scenarios.csv"
    model_before = BaselineOPFModel()
    params_before = [p.clone().detach() for p in model_before.parameters()]
    model_after = train_baseline_model(csv_file, epochs=5)
    params_after = [p for p in model_after.parameters()]
    # Check that at least one parameter has changed.
    changed = any(torch.any(p_after != p_before) for p_before, p_after in zip(params_before, params_after))
    assert changed, "Model parameters did not change after training." 