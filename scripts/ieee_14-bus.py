from src.pypsa_data_generation import generate_opf_scenarios
from src.baseline_opf import train_baseline_model
from src.train_gnn import train_gnn
from src.evaluate_gnn import save_model, load_model, evaluate_model

def run_ieee_14bus_pipeline():
    print("=== Step 1: Generate OPF Scenarios ===")
    scenario_csv = generate_opf_scenarios(num_scenarios=10, load_variation=0.3)

    print("\n=== Step 2: Run Baseline Model ===")
    train_baseline_model(scenario_csv, epochs=5, batch_size=4, learning_rate=0.01)

    print("\n=== Step 3: Train the Physics-Informed GNN ===")
    gnn_model = train_gnn(num_epochs=5, learning_rate=0.01)

    print("\n=== Step 4: Evaluate & Save Model ===")
    save_model(gnn_model, "model_checkpoint.pth")
    loaded_model = load_model("model_checkpoint.pth")
    results = evaluate_model(loaded_model, test_csv=scenario_csv)
    print("\nEvaluation Results:")
    for r in results:
        print(r)

    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    # conda activate gnn_opf_env
    run_ieee_14bus_pipeline()