#!/usr/bin/env python3
"""
Main pipeline script for the GNN-OPF project.
This script runs the complete pipeline from data generation to model evaluation.
"""

import logging
import os
from pathlib import Path

from gnn_opf.pypsa_data_generation import generate_opf_scenarios
from gnn_opf.baseline_opf import train_baseline_model
from gnn_opf.train_gnn import train_gnn
from gnn_opf.evaluate_gnn import save_model, load_model, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def run_ieee_14bus_pipeline(
    num_scenarios=10,
    load_variation=0.3,
    epochs=5,
    batch_size=4,
    learning_rate=0.01
):
    """
    Run the complete IEEE 14-bus pipeline.
    
    Args:
        num_scenarios (int): Number of OPF scenarios to generate
        load_variation (float): Load variation factor for scenario generation
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization
    """
    try:
        # Ensure data directory exists
        data_dir = ensure_data_dir()
        
        logger.info("=== Step 1: Generate OPF Scenarios ===")
        scenario_csv = generate_opf_scenarios(
            num_scenarios=num_scenarios,
            load_variation=load_variation
        )
        logger.info(f"Scenarios saved to: {scenario_csv}")

        logger.info("\n=== Step 2: Run Baseline Model ===")
        baseline_model = train_baseline_model(
            scenario_csv,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        logger.info("Baseline model training completed")

        logger.info("\n=== Step 3: Train the Physics-Informed GNN ===")
        gnn_model = train_gnn(
            num_epochs=epochs,
            learning_rate=learning_rate
        )
        logger.info("GNN model training completed")

        logger.info("\n=== Step 4: Evaluate & Save Model ===")
        model_path = data_dir / "model_checkpoint.pth"
        save_model(gnn_model, str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
        loaded_model = load_model(str(model_path))
        results = evaluate_model(loaded_model, test_csv=scenario_csv)
        
        logger.info("\nEvaluation Results:")
        for result in results:
            logger.info(f"Scenario {result['scenario']}: "
                       f"Predicted={result['predicted_total']:.4f}, "
                       f"Actual={result['target_total']:.4f}, "
                       f"Error={result['error']:.4f}")

        logger.info("\n=== Pipeline Completed Successfully ===")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    run_ieee_14bus_pipeline()