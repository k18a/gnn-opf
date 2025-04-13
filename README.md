# GNN-OPF Project

A comprehensive project for developing a physics-informed Graph Neural Network (GNN) approach to Optimal Power Flow (OPF). This repository integrates multiple components—from network configuration and scenario generation to baseline modeling, graph conversion, GNN training, inference, and evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Project Features](#project-features)
3. [Project Structure](#project-structure)
4. [Installation & Environment Setup](#installation--environment-setup)
5. [Usage Instructions](#usage-instructions)
   - [Configuration File](#configuration-file)
   - [Running Data Generation](#running-data-generation)
   - [Baseline Model](#baseline-model)
   - [Graph Data Conversion](#graph-data-conversion)
   - [Physics-Informed GNN Model](#physics-informed-gnn-model)
   - [Inference Pipeline](#inference-pipeline)
   - [Training Pipeline](#training-pipeline)
   - [Evaluation & Model Persistence](#evaluation--model-persistence)
6. [Running Tests](#running-tests)
7. [Future Work](#future-work)
8. [Credits & Acknowledgments](#credits--acknowledgments)

## Overview

This project builds an end-to-end pipeline to approximate Optimal Power Flow (OPF) in power networks using a physics-informed Graph Neural Network. Key aspects include:
- A configuration file based on the IEEE 14-bus test case.
- Modules for setting up a PyPSA network, generating OPF scenarios, converting network data into graph format using PyTorch Geometric, and implementing a simple physics-informed GNN.
- Baseline models (e.g., a simple MLP) for comparative purposes.
- Training, inference, and evaluation pipelines to allow model learning, persistence, and performance assessment.

## Project Features

- **Network Configuration:**  
  Uses a YAML file (`config/ieee_14-bus_config.yaml`) that defines buses, lines, and generators for an IEEE 14-bus test case.

- **PyPSA Setup Module:**  
  A module (`src/pypsa_setup.py`) that loads the configuration file and creates a PyPSA network.

- **Scenario Generation:**  
  A data generation module (`src/pypsa_data_generation.py`) that simulates multiple OPF scenarios, perturbs loads, solves the DC OPF, and writes key metrics (e.g., total_cost) to a CSV file.

- **Baseline Model:**  
  A simple MLP model (`src/baseline_opf.py`) that predicts the total OPF cost using the scenario number as a feature.

- **Graph Conversion:**  
  A module (`src/data/graph_data.py`) that converts the PyPSA network into a PyTorch Geometric `Data` object, with nodes representing buses and edges representing lines.

- **Physics-Informed GNN Model:**  
  A module (`src/gnn_opf.py`) that implements a GNN with two GCNConv layers (with a ReLU in between) and includes a basic physics penalty function.

- **Inference Pipeline:**  
  A module (`src/inference_pipeline.py`) that loads the network configuration, converts it to a graph, runs the GNN model for inference, and computes a physics penalty.

- **Training Pipeline:**  
  A training module (`src/train_gnn.py`) that reads scenario data from CSV, adjusts network loads for each scenario, converts the network to graph data, and optimizes the GNN model using mean squared error (MSE) loss combined with a physics penalty.

- **Evaluation & Model Persistence:**  
  A module (`src/evaluate_gnn.py`) that provides functions to save and load the trained model and evaluate its performance over test scenarios.

## Project Structure

gnn_opf_project/
├── config/
│   └── ieee_14-bus_config.yaml   # IEEE 14-bus network configuration
├── data/
│   └── generated_opf_scenarios.csv  # OPF scenario data generated by the system
├── docs/                         # Documentation (future updates)
├── notebooks/                    # Jupyter notebooks for exploration
├── scripts/                      # Helper scripts for automation (future work)
├── src/
│   ├── init.py               # Makes src/ a Python package
│   ├── baseline_opf.py           # MLP-based baseline model for OPF
│   ├── evaluate_gnn.py           # Model evaluation, saving, and loading functions
│   ├── gnn_opf.py                # Physics-informed GNN model implementation
│   ├── inference_pipeline.py     # Inference pipeline using the GNN model
│   ├── pypsa_data_generation.py  # Data generation module for OPF scenarios
│   └── pypsa_setup.py            # Loads config file and creates a PyPSA network
├── tests/
│   ├── test_baseline_opf.py     # Tests for the baseline model
│   ├── test_evaluate_gnn.py     # Tests for model evaluation and persistence functions
│   ├── test_graph_data.py       # Tests for the graph conversion module
│   ├── test_inference_pipeline.py  # Tests for the inference pipeline
│   ├── test_pypsa_data_generation.py # Tests for the OPF scenario generation module
│   └── test_pypsa_setup.py      # Tests for the PyPSA setup module
├── environment.yml               # Conda environment specification
└── README.md                     # Project overview and instructions (this file)

## Installation & Environment Setup

1. **Clone the Repository:**  
   If this project is hosted on GitHub, clone it using:
   ```bash
   git clone <repository_url>
   cd gnn_opf_project

2. **Create Conda Environment:**
Make sure Conda is installed. Then run:

```bash
conda create -n gnn_opf_env python=3.9
conda activate gnn_opf_env
```

Verify the Python version with:

```bash
python --version
```

(It should output Python 3.9.x)

3. **Install Additional Dependencies (if needed):**
The primary dependencies are listed in environment.yml. Additional packages (e.g., for NLP or visualization) can be installed later via conda or pip.

## Usage Instructions

Configuration File
- Location: config/ieee_14-bus_config.yaml
- Purpose: Contains the network parameters for the IEEE 14-bus test case (buses, lines, generators).
- Modification: Do not change the keys or structure unless you are updating the test network.

Running Data Generation
- Module: src/pypsa_data_generation.py
- Usage:
From the project root, run:

```bash
python src/pypsa_data_generation.py
```

This will generate OPF scenarios and save the results to data/generated_opf_scenarios.csv.

Baseline Model
- Module: src/baseline_opf.py
- Usage:
You can run the baseline model script directly:

```bash
python src/baseline_opf.py
```

This script trains the simple MLP model to predict total_cost based on the scenario number.

Graph Data Conversion
- Module: src/data/graph_data.py
- Usage:
This module is integrated within other parts of the project. It converts a PyPSA network to a PyTorch Geometric Data object for use by the GNN models.

Physics-Informed GNN Model
- Module: src/gnn_opf.py
- Usage:
This module contains the implementation of the GNN model (using two GCNConv layers) and a physics penalty function. It is used by both the inference and training pipelines.

Inference Pipeline
- Module: src/inference_pipeline.py
- Usage:
Run the inference pipeline to load the network, convert it to a graph, run the GNN, and compute the physics penalty:

```bash
python src/inference_pipeline.py
```

Training Pipeline
- Module: src/train_gnn.py
- Usage:
To train the Physics-Informed GNN model on the generated OPF scenario data, run:

```bash
python src/train_gnn.py
```

You can adjust training parameters such as the number of epochs and learning rate inside the script.

Evaluation & Model Persistence
- Module: src/evaluate_gnn.py
- Usage:
To evaluate the trained model, save/load it, and review evaluation metrics, run:

```bash
python src/evaluate_gnn.py
```

This module saves the model checkpoint to model_checkpoint.pth and loads it back for evaluation.

Running Tests

All tests have been written using pytest. To run all tests, execute:

```bash
pytest
```

Ensure you are in the project root and the conda environment gnn_opf_env is active.

Future Work

Planned improvements and next steps:
- Refine the Physics Penalty Function: Implement more realistic physical constraints.
- Advanced NLP Preprocessing: Enhance memory retrieval by extracting key phrases from queries.
- Model Scaling: Test the system on larger, more complex power networks.
- Integration with Chat and AdaAgent: Develop additional modules to incorporate the GNN-OPF model into a full conversational agent (AdaAgent) for intelligent decision-making.
- Enhanced Evaluation Metrics: Incorporate metrics beyond MSE, such as feasibility checks and constraint satisfaction.

Credits & Acknowledgments
- PyPSA: For providing a robust power system analysis library.
- PyTorch & PyTorch Geometric: For enabling state-of-the-art deep learning and graph network modeling.
- Community Contributions: Thanks to all contributors and researchers whose insights shaped this project.

⸻

This README provides a comprehensive overview of what has been implemented so far and instructions on how to run and test the project. As the project evolves, additional features and modules will be documented in subsequent updates.

## Development

For development, you might want to install additional dependencies:
```bash
pip install -e ".[dev]"
```

