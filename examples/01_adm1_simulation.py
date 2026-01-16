"""
ADM1 Simulation Example
=======================

This script demonstrates ADM1 model simulation using the unified MechanisticModel interface.

Workflow:
1. Load Data: Use ACoD preprocessing to generate influent from feedstock ratios.
2. Initialize: Use ADM1Model (MechanisticModel).
3. Simulate: Use unified simulate() method.
4. Evaluate: Use unified evaluate() method.

New in v0.2.0: Uses simplified API with top-level imports.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import openad_lib as openad
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def run_pipeline():
    print("="*60)
    print("ADM1 Integrated Pipeline (Updated Structure)")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. Configuration & Paths
    # -------------------------------------------------------------------------
    base_data_path = src_path / 'openad_lib' / 'data'
    ratios_file = base_data_path / 'feedstock' / 'Feed_Data.csv'
    measured_data_file = base_data_path / 'Biogas_Plant_Outputs.csv'
    results_file = 'ADM1_Final_results.csv'
    
    if not ratios_file.exists():
        print(f"Error: Ratios file not found at {ratios_file}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Input Characterization (ACoD)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Generating Influent Data (ACoD) ---")
    # Generate influent from feedstock data
    print("\nGenerating influent parameters...")
    influent_df = openad.acod.generate_influent_data(str(ratios_file))
    print(f"Generated influent data: {influent_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Dynamic Simulation
    # -------------------------------------------------------------------------
    # Initialize ADM1 model (using simplified API)
    print("\nInitializing ADM1 model...")
    model = openad.ADM1Model()
    
    # Run simulation using new API
    simulation_output = model.simulate(influent_df)
    
    # Extract results
    df_qgas = simulation_output['q_gas']
    qgas_sim = df_qgas['q_gas'].values
    
    # Save results
    print(f"\nSaving results to {results_file}...")
    df_qgas.to_csv(results_file, index=False)
    print(f"Saved gas flow results to {results_file}")

    # -------------------------------------------------------------------------
    # 4. Evaluation & Visualization
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Evaluation & Visualization ---")
    
    has_measured = True
    if measured_data_file.exists():
        print(f"Loading measured data from {measured_data_file}")
        try:
            biogas_data = pd.read_csv(measured_data_file)
            
            # Determine target column
            if 'q_gas' in biogas_data.columns:
                target_col = 'q_gas'
            elif 'Biogas (m3/day)' in biogas_data.columns:
                target_col = 'Biogas (m3/day)'
            elif 'Biogas' in biogas_data.columns:
                target_col = 'Biogas'
            else:
                target_col = None
            
            if target_col:
                y_true = biogas_data[target_col].values
                # Align lengths
                min_len = min(len(y_true), len(qgas_sim))
                y_true_aligned = y_true[:min_len]
                y_pred_aligned = qgas_sim[:min_len]
                time_aligned = biogas_data['time'].values[:min_len] if 'time' in biogas_data.columns else np.arange(min_len)

                # Compute Metrics using unified module
                metrics = compute_metrics(y_true_aligned, y_pred_aligned)
                print_metrics(metrics, title="Model Performance (Total Biogas)")
                has_measured = True
            else:
                print("Warning: Measured biogas column not found.")
        except Exception as e:
            print(f"Warning: Could not process measured data: {e}")
    else:
        print("Measured data file not found. Skipping comparison.")

    # Plotting using unified system
    images_dir = current_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True)
    plot_file = images_dir / 'adm1_comparison_plot.png'
    
    if has_measured:
        openad.plots.plot_predictions(
            y_true=y_true_aligned,
            y_pred=y_pred_aligned,
            x=time_aligned,
            title="ADM1 Simulation Results",
            xlabel="Time (days)",
            ylabel="Biogas Production Rate (m³/day)",
            save_path=plot_file,
            show=False
        )
    else:
        openad.plots.plot_predictions(
            y_true=qgas_sim,  # No measured data, just show predictions
            y_pred=qgas_sim,
            x=t,
            title="ADM1 Simulation Results",
            xlabel="Time (days)",
            ylabel="Biogas Production Rate (m³/day)",
            save_path=plot_file,
            show=False
        )
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    run_pipeline()
