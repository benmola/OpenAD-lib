"""
AM2 Model Simulation Example
============================

This script demonstrates the AM2 simplified model using the unified MechanisticModel interface.

Workflow:
1. Load Data: Use load_sample_data() for built-in sample data.
2. Initialize: Use AM2Model (MechanisticModel).
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
from openad_lib.utils.metrics import print_metrics

def run_simulation():
    print("=" * 60)
    print("AM2 Model Simulation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Load Data Using Library Utilities  
    # -------------------------------------------------------------------------
    # Load sample data directly - no file paths needed!
    am2_df = openad.load_sample_data('am2_lab')
    print(f"Loaded {len(am2_df)} rows of sample data")

    # Initialize AM2 model (using simplified API)
    print("\nInitializing AM2 model...")
    model = openad.AM2Model()
    
    # Display default parameters (legacy access still works)
    print("\nDefault AM2 Parameters:")
    print(f"  mu1max (m1): {model.params.m1} 1/d")
    print(f"  K1:          {model.params.K1} g COD/L")
    print(f"  mu2max (m2): {model.params.m2} 1/d")
    print(f"  Ki:          {model.params.Ki} g COD/L")
    print(f"  K2:          {model.params.K2} g COD/L")
    
    # Load data into model directly from DataFrame
    print("\nLoading data into model...")
    model.load_data_from_dataframe(
        am2_df,
        S1in_col='SCODin',  # Map SCOD input
        S1out_col='SCODout', # Map SCOD output
        S2out_col='VFAout',  # Map VFA output
        Q_col='Biogas'       # Map Biogas flow
    )
    
    # Simulate (Unified API)
    print("\nRunning AM2 simulation...")
    results = model.simulate()
    
    # Extract results
    time = results['time']
    S1 = results['S1']
    S2 = results['S2']
    Q = results['Q']
    
    
    
    # -------------------------------------------------------------------------
    # 2. Plotting using unified system
    # -------------------------------------------------------------------------
    print("\nGenerating plots...")
    
    # Get measured data from model
    measured = model.data
    meas_time = measured['time'].values
    
    # Get measured outputs
    Y_measured = np.column_stack([
        measured['S1out'].values,
        measured['S2out'].values,
        measured['Q'].values
    ])
    
    # Simulated data (already aligned - simulate uses same time points)
    Y_simulated = np.column_stack([S1, S2, Q])
    


    # Evaluate if measured data exists (it does in this sample)
    print("\nEvaluation Metrics:")
    metrics = model.evaluate()
    print_metrics(metrics, title="AM2 Model Performance")
    



    # Save plot - auto-save enabled!
    openad.plots.plot_multi_output(
        y_true=Y_measured,
        y_pred=Y_simulated,
        x=meas_time,
        output_names=['Substrate S1 (g COD/L)', 'VFA S2 (g COD/L)', 'Methane Flow Q (L/d)'],
        title="AM2 Model Simulation",
        xlabel='Time (days)',
        save_plot=True,
        show=False
    )

if __name__ == "__main__":
    run_simulation()
