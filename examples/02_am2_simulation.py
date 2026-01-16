"""
AM2 Model Simulation Example
============================

This script demonstrates the AM2 simplified model using the unified MechanisticModel interface.

Workflow:
1. Load Data: Use model.load_data() method.
2. Initialize: Use AM2Model (MechanisticModel).
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
    import openad_lib as oad
    from openad_lib.utils.metrics import print_metrics
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def run_simulation():
    print("=" * 60)
    print("AM2 Model Simulation")
    print("=" * 60)
    
    # Path setup
    base_data_path = src_path / 'openad_lib' / 'data'
    am2_data_path = base_data_path / 'sample_AM2_Lab_data.csv'
    
    if not am2_data_path.exists():
        print(f"Error: Data file not found at {am2_data_path}")
        return

    # Initialize AM2 model (using simplified API)
    print("\nInitializing AM2 model...")
    model = oad.AM2Model()
    
    # Display default parameters (legacy access still works)
    print("\nDefault AM2 Parameters:")
    print(f"  µ1max (m1): {model.params.m1} d⁻¹")
    print(f"  K1:         {model.params.K1} g COD/L")
    print(f"  µ2max (m2): {model.params.m2} d⁻¹")
    print(f"  Ki:         {model.params.Ki} g COD/L")
    print(f"  K2:         {model.params.K2} g COD/L")
    
    # Load data (Unified API)
    print("\nLoading AM2 data...")
    model.load_data(str(am2_data_path))
    
    # Simulate (Unified API)
    print("\nRunning AM2 simulation...")
    results = model.simulate()
    
    # Extract results
    time = results['time']
    S1 = results['S1']
    S2 = results['S2']
    Q = results['Q']
    
    # Evaluate if measured data exists (it does in this sample)
    print("\nEvaluation Metrics:")
    # Note: evaluate() automatically uses loaded measured data if columns match
    # 'S1out', 'S2out', 'Q' are expected in measured data for AM2 evaluation
    try:
        metrics = model.evaluate()
        print_metrics(metrics, title="AM2 Model Performance")
    except Exception as e:
        print(f"Could not evaluate metrics: {e}")
    
    # Plotting
    print("\nGenerating plots...")
    plt.style.use('bmh')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot S1
    ax1.plot(time, S1, label='Simulated S1', color='tab:blue', linewidth=2)
    if 'S1out' in model.data.columns:
        ax1.plot(model.data['time'], model.data['S1out'], 'o', label='Measured S1', color='tab:blue', alpha=0.6)
    ax1.set_ylabel('Substrate S1 (g COD/L)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot S2
    ax2.plot(time, S2, label='Simulated S2', color='tab:orange', linewidth=2)
    if 'S2out' in model.data.columns:
        ax2.plot(model.data['time'], model.data['S2out'], 'o', label='Measured S2', color='tab:orange', alpha=0.6)
    ax2.set_ylabel('VFA S2 (g COD/L)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot Q
    ax3.plot(time, Q, label='Simulated Q', color='tab:green', linewidth=2)
    if 'Q' in model.data.columns:
        ax3.plot(model.data['time'], model.data['Q'], 'o', label='Measured Q', color='tab:green', alpha=0.6)
    ax3.set_ylabel('Methane Flow Q (L/d)')
    ax3.set_xlabel('Time (days)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('AM2 Simulation Results', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    images_dir = current_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True)
    plot_file = images_dir / 'am2_simulation.png'
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    run_simulation()
