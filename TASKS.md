# Task List for Project Setup (Already in gnn_opf_project)

1. **Initialize Git and Create Basic Files**
   - [x] Run `git init` in the current folder. _(Completed: Initialized git repository in the project folder)_
   - [x] Create a file named `README.md` with the exact content below: _(Completed: Created README.md with specified content)_
     ```
     # GNN-OPF Project
     This repository contains code for a Graph Neural Network approach to Optimal Power Flow.
     ```
   - [x] Create a file named `.gitignore` with the exact content below: _(Completed: Created .gitignore with specified content)_
     ```
     __pycache__/
     .ipynb_checkpoints/
     *.pyc
     *.pyo
     .DS_Store
     data/
     ```

2. **Create the Required Folders**
   - [x] Create the following folders in the current project folder: _(Completed: Created all required directories using mkdir command)_
     - `src/`
     - `tests/`
     - `config/`
     - `data/`
     - `notebooks/`
     - `docs/`
     - `scripts/`

3. **Create a Conda Environment**
   - [ ] Open a terminal in the current folder and run the following commands:
     ```
     conda create -y -n gnn_opf_env python=3.9
     conda activate gnn_opf_env
     ```
   - [ ] Verify that the environment is using Python 3.9 by running:  
     `python --version`

4. **Define Environment Dependencies**
   - [x] Create a file named `environment.yml` in the current folder with the exact content below: _(Completed: Created environment.yml with specified dependencies)_
     ```yaml
     name: gnn_opf_env
     channels:
       - pytorch
       - conda-forge
       - defaults
     dependencies:
       - python=3.9
       - pip
       - pytest
       - pypsa
       - pytorch
       - pyg
       - requests
       - jupyter
       - pip:
         - torch-geometric
     ```

5. **Install the Environment**
   - [ ] Run the following commands in the terminal:
     ```
     conda env create -f environment.yml
     conda activate gnn_opf_env
     ```
   - [ ] Verify that Python is version 3.9 by executing:  
     `python --version`

6. **Commit the Initial Setup**
   - [ ] In the terminal, run:
     ```
     git add .
     git commit -m "Initial project setup: environment, folder structure, README, .gitignore, and environment.yml"
     ``` 