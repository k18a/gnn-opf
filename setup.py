from setuptools import setup, find_packages

setup(
    name="gnn_opf",
    version="0.1.0",
    description="GNN-based Optimal Power Flow solver",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pypsa",
        "torch",
        "torch-geometric",
        "pytest",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ]
    },
    python_requires=">=3.9",
) 