"""
OPENAD-LIB IMPROVED STRUCTURE DEMO
==================================

This script demonstrates the new unified structure of the openad_lib package.
It showcases:
1. Unified MechanisticModel interface (ADM1Model)
2. Unified Metrics (compute_metrics)
3. New Calibration Framework (ADM1Calibrator)
4. Consistent API usage across models

New in v0.2.0: Uses simplified API with top-level imports.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Imports from new structure (using simplified API)
try:
    import openad_lib as oad
    from openad_lib.utils.metrics import print_metrics
except ImportError as e:
    print(f"Error importing library: {e}")
    sys.exit(1)

def run_demo():
    print("="*80)
    print("OPENAD-LIB IMPROVED STRUCTURE DEMO")
    print("="*80)
    
    # Paths to data
    base_data_path = src_path / 'openad_lib' / 'data'
    feed_data_path = base_data_path / 'feedstock' / 'Feed_Data.csv'
    measured_data_path = base_data_path / 'Biogas_Plant_Outputs.csv'
    
    # ----------------------------------------------------------------
    # 1. Unified Interface: ADM1 Model
    # ----------------------------------------------------------------
    print("\n\n1️⃣  UNIFIED INTERFACE: ADM1 MODEL")
    print("-" * 40)
    print("Initializing ADM1Model (using simplified API)...")
    adm1 = oad.ADM1Model()
    
    # Demonstrate load_data / simulation flow
    print("Generating influent data from Feed_Data.csv...")
    if feed_data_path.exists():
        influent_df = oad.acod.generate_influent_data(str(feed_data_path))
        
        print("\nRunning simulation...")
        # Unified simulate() method returns dictionary of results
        sim_results = adm1.simulate(influent_df)
        
        # Access results
        df_qgas = sim_results['q_gas']
        print(f"Simulation complete. Steps: {len(df_qgas)}")
    else:
        print("Feed data not found. Skipping simulation.")
        return

    # ----------------------------------------------------------------
    # 2. Unified Metrics & Evaluation
    # ----------------------------------------------------------------
    print("\n\n2️⃣  UNIFIED METRICS & EVALUATION")
    print("-" * 40)
    
    if measured_data_path.exists():
        measured_df = pd.read_csv(measured_data_path)
        
        # Determine correct column for biogas flow
        target_col = None
        if 'q_gas' in measured_df.columns:
            target_col = 'q_gas'
        elif 'Biogas (m3/day)' in measured_df.columns:
            target_col = 'Biogas (m3/day)'
        elif 'Biogas' in measured_df.columns:
            print("Note: Using 'Biogas' column for q_gas comparison")
            target_col = 'Biogas'
            
        if target_col:
            print("Evaluating performance...")
            # Prepare data alignment
            y_pred = df_qgas['q_gas'].values
            y_true = measured_df[target_col].values
            
            # Truncate to matching length
            n = min(len(y_pred), len(y_true))
            y_pred = y_pred[:n]
            y_true = y_true[:n]
            
            # Unified evaluate() usage
            # Manually calling evaluate on model (or using compute_metrics directly)
            metrics = adm1.evaluate(y_true, y_pred)
            print_metrics(metrics, title="ADM1 Baseline Performance (q_gas)")
        else:
            print("Warning: No suitable biogas column found in Biogas_Plant_Outputs.csv")
            print(f"Available columns: {measured_df.columns.tolist()}")

    # ----------------------------------------------------------------
    # 3. New Feature: ADM1 Calibration
    # ----------------------------------------------------------------
    print("\n\n3️⃣  NEW FEATURE: ADM1 CALIBRATION")
    print("-" * 40)
    
    if measured_data_path.exists() and feed_data_path.exists():
        print("Setting up ADM1 Calibrator (Optuna-based)...")
        
        # Configure calibrator (using simplified API)
        calibrator = oad.ADM1Calibrator(
            model=adm1,
            measurement_data=pd.read_csv(measured_data_path),
            influent_data=influent_df
        )
        
        # Define parameters to calibrate
        params_to_tune = ['k_hyd', 'k_m_ac', 'K_S_ac']
        
        # Construct bounds for selected parameters
        # Access default bounds from the calibrator instance
        subset_bounds = {k: calibrator.default_bounds[k] for k in params_to_tune}
        
        # Run calibration (short run for demo)
        print("Calibrating 3 parameters for 5 trials (demo mode)...")
        best_params = calibrator.calibrate(
            target_outputs=['q_gas'], # Calibrate against gas flow
            param_bounds=subset_bounds,
            n_trials=5
        )
        
        print("\nCalibration Results:")
        # The calibrate method returns a dictionary containing 'best_value', 'best_params', etc.
        # We named the variable 'best_params' but it holds the full results dict.
        print(f"Best RMSE: {best_params['best_value']:.4f}")
        print("Best Parameters:")
        for k, v in best_params['best_params'].items():
            print(f"  - {k}: {v:.4f}")
            
        # Update model with best params
        print("\nUpdating model with best parameters...")
        adm1.update_params(best_params['best_params'])
        
        # Re-run verification
        print("Verifying improvement...")
        new_results = adm1.simulate(influent_df)
        if target_col:
             new_qgas = new_results['q_gas']['q_gas'].values[:n]
             new_metrics = adm1.evaluate(y_true, new_qgas)
             print_metrics(new_metrics, title="ADM1 Calibrated Performance (q_gas)")

    # ----------------------------------------------------------------
    # 4. Consistency Check
    # ----------------------------------------------------------------
    print("\n\n4️⃣  CONSISTENCY CHECK: AM2 MODEL")
    print("-" * 40)
    print("Initializing AM2Model (uses same Base structure, simplified API)...")
    am2 = oad.AM2Model()
    
    # Verify inheritance
    from openad_lib.models.base import MechanisticModel
    is_mech = isinstance(am2, MechanisticModel)
    print(f"AM2 inherits from MechanisticModel: {is_mech}")
    print("AM2Model exposes uniform API: simulate(), update_params(), evaluate()")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_demo()
