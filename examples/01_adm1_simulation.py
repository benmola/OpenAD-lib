"""
Example: ADM1 Model Simulation

This script demonstrates how to use the ADM1 mechanistic model for anaerobic digestion.
It loads sample data, runs the simulation, and plots the results.
"""

import os
import matplotlib.pyplot as plt
from openad_lib.models.mechanistic import ADM1Model

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')

print("=" * 60)
print("ADM1 Model Simulation")
print("=" * 60)

# Check if data files exist
influent_path = os.path.join(DATA_DIR, 'sample_ADM1_influent_data.csv')
initial_path = os.path.join(DATA_DIR, 'sample_initial_state.csv')

if os.path.exists(influent_path) and os.path.exists(initial_path):
    print("\nInitializing ADM1 model...")
    model = ADM1Model()
    
    # Configure reactor parameters
    model.params.V_liq = 10000  # m³
    model.params.V_gas = 600    # m³
    
    print(f"  Reactor volume: {model.params.V_liq} m³")
    print(f"  Gas headspace: {model.params.V_gas} m³")
    
    # Load data
    print("\nLoading influent and initial state data...")
    model.load_data(
        influent_path=influent_path,
        initial_state_path=initial_path,
        influent_sheet="Influent_ADM1_COD_Based"
    )
    
    # Run simulation
    print("\nRunning ADM1 simulation...")
    results = model.run(solver_method="BDF", verbose=True)
    
    # Plot results
    print("\nGenerating plots...")
    model.plot_results()
    
else:
    print(f"Error: Data files not found in {DATA_DIR}")
    print("Please ensure sample_ADM1_influent_data.csv and sample_initial_state.csv exist.")
