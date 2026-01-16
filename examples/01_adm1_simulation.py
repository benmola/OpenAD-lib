"""
ADM1 Simulation Example
=======================

This script demonstrates ADM1 model simulation using the unified MechanisticModel interface.

Workflow:
1. Load Data: Use ACoD preprocessing to generate influent from feedstock ratios.
2. Initialize: Use ADM1Model (MechanisticModel).
3. Simulate: Use unified simulate() method.
4. Evaluate: Use unified evaluate() method.

New in v0.2.0: Uses simplified API with top-level imports and load_sample_data().
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for development
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import openad_lib as openad

def run_pipeline():
    print("="*60)
    print("ADM1 Integrated Pipeline (Updated Structure)")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. Load Data Using Library Utilities
    # -------------------------------------------------------------------------
    # Use built-in sample data loaders - clean API, no hardcoded paths!
    feedstock_data = openad.load_sample_data('feedstock')
    biogas_data = openad.load_sample_data('biogas')

    # -------------------------------------------------------------------------
    # 2. Input Characterization (ACoD)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Generating Influent Data (ACoD) ---")
    # Generate influent from feedstock data - accepts DataFrame directly!
    influent_df = openad.acod.generate_influent_data(feedstock_data)
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
    #print(f"\nSaving results to {results_file}...")
    #df_qgas.to_csv(results_file, index=False)
    #print(f"Saved gas flow results to {results_file}")

    # -------------------------------------------------------------------------
    # 4. Evaluation & Visualization
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Evaluation & Visualization ---")
    
    target_col = 'Biogas (m3/day)'    
    y_true = biogas_data[target_col].values
    
    # Align lengths
    min_len = min(len(y_true), len(qgas_sim))
    y_true_aligned = y_true[:min_len]
    y_pred_aligned = qgas_sim[:min_len]
    time_aligned = biogas_data['time'].values[:min_len] if 'time' in biogas_data.columns else np.arange(min_len)

    # Compute Metrics using unified API
    metrics = openad.utils.metrics.compute_metrics(y_true_aligned, y_pred_aligned)
    openad.utils.metrics.print_metrics(metrics, title="Model Performance (Total Biogas)")

    # Plotting using unified system - auto-save enabled!
    openad.plots.plot_predictions(
        y_true=y_true_aligned,
        y_pred=y_pred_aligned,
        x=time_aligned,
        title="ADM1 Simulation Results",
        xlabel="Time (days)",
        ylabel="Biogas Production Rate (m3/day)",
        save_plot=True,
        show=True
    )

if __name__ == "__main__":
    run_pipeline()
