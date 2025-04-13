import os
import csv
import pytest
from src.pypsa_data_generation import generate_opf_scenarios

def test_generate_opf_scenarios():
    output_file = generate_opf_scenarios(num_scenarios=5)
    # Check that the output CSV file exists.
    assert os.path.exists(output_file), "CSV output file does not exist."
    
    # Open the CSV file and check its contents.
    with open(output_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        assert "total_cost" in header, "'total_cost' column not found in CSV header."
        # Ensure there's at least one data row.
        data_row = next(reader, None)
        assert data_row is not None, "No data rows found in CSV output." 