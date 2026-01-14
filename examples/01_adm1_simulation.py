"""
Integrated ADM1 Simulation Pipeline
===================================

This script demonstrates the complete workflow for ADM1 simulation:
1. Input Characterization (ACoD): Generates influent state variables from substrate ratios.
2. Dynamic Simulation: Runs the ADM1 model using the generated input.
3. Visualization: Compares simulated biogas production with measured plant data.

Usage:
    python examples/08_integrated_adm1_pipeline.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from openad_lib.preprocessing import acod
    from openad_lib.models.mechanistic.adm1_model import ADM1Model
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def run_pipeline():
    print("="*60)
    print("ADM1 Integrated Pipeline")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. Configuration & Paths
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    project_root = os.path.join(base_dir, '..')
    
    # Input files from My-Codes
    ratios_file = os.path.join(project_root, 'src', 'openad_lib', 'data','feedstock',  'Feed_Data.csv')
    measured_data_file = os.path.join(project_root, 'src', 'openad_lib', 'data', 'Biogas_Plant_Outputs.csv')
    

    results_file = 'ADM1_Final_results.csv'
    
    if not os.path.exists(ratios_file):
        print(f"Error: Ratios file not found at {ratios_file}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Input Characterization (ACoD)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Generating Influent Data (ACoD) ---")
    influent_df = acod.generate_influent_data(ratios_file)
    print("Influent data generated successfully.")
    print(f"Shape: {influent_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Dynamic Simulation
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Running ADM1 Simulation ---")
    model = ADM1Model()
    
    # Run simulation
    # Note: The influent_df typically contains a 'time' column or index.
    # The ADM1Model expects specific column names which ACoD provides.
    
    simulation_output = model.simulate(influent_df)
    
    # Extract results
    df_res = simulation_output['results']
    df_qgas = simulation_output['q_gas']


    # Save results
    print(f"\nSaving results to {results_file}...")


    if results_file.endswith('.csv'):
        try:
            with pd.DataFrame(results_file) as writer:
                df_res.to_csv(writer, index=False)
                df_qgas.to_csv(writer, index=False)
        except Exception as e:
            print(f"Error writing CSV file: {e}")
    else:
        # Fallback for CSV or other formats
        # Write main results to the specified file
        df_res.to_csv(results_file, index=False)
        
        # Write auxiliary outputs to separate files
        base, ext = os.path.splitext(results_file)
        qgas_file = f"{base}_qgas{ext}"

        
        df_qgas.to_csv(qgas_file, index=False)
        print(f"Saved main results to {results_file}")
        print(f"Saved gas flow to {qgas_file}")


    # -------------------------------------------------------------------------
    # 4. Visualization & Comparison
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Visualization ---")
    
    if os.path.exists(measured_data_file):
        print(f"Loading measured data from {measured_data_file}")
        try:
            biogas_data = pd.read_csv(measured_data_file)
            has_measured = True
        except Exception as e:
            print(f"Warning: Could not read measured data: {e}")
            has_measured = False
    else:
        print("Measured data file not found. Skipping comparison.")
        has_measured = False

    # Plotting
    plt.style.use('bmh')
    plt.figure(figsize=(12, 8))
    
    # Time vector
    t = df_qgas['time'] if 'time' in df_qgas.columns else df_qgas.index
    qgas_sim = df_qgas['q_gas']

    if has_measured:
        # Assuming measured data has 'time' and 'Biogas (m3/day)' columns
        # Check columns
        if 'time' in biogas_data.columns and 'Biogas (m3/day)' in biogas_data.columns:
            plt.plot(biogas_data['time'], biogas_data['Biogas (m3/day)'], 
                     label='Measured Data', 
                     linestyle='--', 
                     color='#2E86C1', 
                     linewidth=3, 
                     alpha=0.8)
        else:
            print("Measured data columns mismatch. Plotting available columns if possible.")
            print(f"Columns: {biogas_data.columns}")

    plt.plot(t, qgas_sim, 
             label='Model Prediction', 
             linestyle='-', 
             color='#E67E22', 
             linewidth=2)

    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    plt.xlabel('Time (days)', fontsize=14, fontweight='bold')
    plt.ylabel('Biogas Production Rate (m³/day)', fontsize=14, fontweight='bold')
    plt.title('Comparison of Measured and Simulated Biogas Production', fontsize=16, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    images_dir = os.path.join(project_root, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    plot_file = os.path.join(images_dir, 'adm1_comparison_plot.png')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    
    # plt.show() # Non-blocking for agent environment

if __name__ == "__main__":
    run_pipeline()
