import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define a simple dataset class that reads the generated CSV file.
class OPFDataset(Dataset):
    def __init__(self, csv_file):
        self.samples = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Read both the scenario number and bus1_load as float features.
                feature1 = float(row["scenario"])
                feature2 = float(row["bus1_load"])
                target = float(row["total_cost"])
                # Combine them into a list of features.
                self.samples.append(([feature1, feature2], target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, target = self.samples[idx]
        # Convert to tensors; model now expects a 2D input.
        return torch.tensor(features, dtype=torch.float), torch.tensor([target], dtype=torch.float)

# Define a simple MLP model.
class BaselineOPFModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super(BaselineOPFModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Define a training function.
def train_baseline_model(csv_file, epochs=50, batch_size=4, learning_rate=0.01):
    dataset = OPFDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BaselineOPFModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for features, target in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Print epoch loss for debugging.
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    return model

# If run as a script, train the model using the generated CSV and print a sample prediction.
if __name__ == "__main__":
    csv_file = "data/generated_opf_scenarios.csv"
    model = train_baseline_model(csv_file)
    # Test a sample prediction using scenario = 1
    test_feature = torch.tensor([[1.0, 0.5]])
    prediction = model(test_feature)
    print(f"Sample prediction for scenario 1: {prediction.item()}") 