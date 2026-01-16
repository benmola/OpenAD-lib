"""
Example: AM2 MPC Control
=======================

This script demonstrates how to use the Model Predictive Controller (MPC)
for the AM2 anaerobic digestion model.

Scenario:
1. Initialize the system at a steady state.
2. Setup the MPC to maximize biogas production.
3. Simulate the closed-loop system for 50 days.
4. The dilution rate (D) is the manipulated variable.
"""

import sys
import os
import numpy as np

import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import openad_lib as openad

def run_control_example():
    print("=" * 60)
    print("AM2 MPC Control Demo: Maximizing Biogas")
    print("=" * 60)
    
    # 1. Setup Parameters
    params = openad.AM2Parameters() # Default parameters
    
    # 2. Initialize Controller
    print("\nInitializing MPC controller...")
    # maximize_biogas mode: minimizing -Q
    controller = openad.AM2MPC(params)
    
    # Configure: 1 day steps, 10 day horizon
    # Slower sampling for biological processes
    sampling_time = 1.0 
    horizon = 10
    
    controller.setup_controller(
        sampling_time=sampling_time,
        horizon=horizon,
        objective_type='maximize_biogas',
        D_max=0.5 # Max dilution rate
    )
    
    # Setup simulator for validation
    controller.setup_simulator(sampling_time=sampling_time)
    
    # 3. Initial Conditions
    # Start from a reasonable operating point
    x0 = np.array([5.0, 1.0, 10.0, 1.0]) # [S1, X1, S2, X2]
    controller.set_initial_state(x0)
    
    # 4. Simulation Loop
    n_days = 50
    
    # Time-varying parameters (Inputs/Disturbances)
    # Let's assume constant inlet load for simplicity, or we could add a step change
    S1in_nominal = 15.0
    pH_nominal = 7.0
    
    # Storage for results
    history = {
        'time': [],
        'D': [],
        'S1': [], 'X1': [], 'S2': [], 'X2': [],
        'Q': []
    }
    
    print(f"\nStarting simulation for {n_days} days...")
    
    t_current = 0.0
    
    for k in range(n_days):
        # Create disturbances (e.g. inlet concentration fluctuation)
        # S1in varies sinusoidally
        S1in_k = S1in_nominal + 5.0 * np.sin(2 * np.pi * k / 20)
        
        # pH stays constant optimum
        pH_k = 7.0
        
        # Run one control step
        # This solves the optimization problem and simulates one step
        u_opt, y_next = controller.run_step(S1in_val=S1in_k, pH_val=pH_k)
        
        # Calculate resulting Q (biogas) for logging
        # Q = k6 * mu2 * X2 * c
        # We need to manually calculate it or get it from simulator aux
        # Quick manual calculation using model logic:
        # Cast to float to handle 1-element arrays from CasADi/numpy
        S2_curr = float(y_next[2])
        X2_curr = float(y_next[3])
        
        # Recalculate kinetics for Q
        p = params
        mu2_base = p.m2 * (S2_curr / ((S2_curr**2)/p.Ki + S2_curr + p.K2))
        pH_factor = np.exp(-4 * ((pH_k - p.pHH)/(p.pHH - p.pHL))**2)
        mu2 = mu2_base * (1.0 - pH_factor)
        Q_curr = float(p.k6 * mu2 * X2_curr * p.c)
        
        # Store data
        history['time'].append(t_current)
        history['D'].append(u_opt)
        history['S1'].append(y_next[0])
        history['X1'].append(y_next[1])
        history['S2'].append(y_next[2])
        history['X2'].append(y_next[3])
        history['Q'].append(Q_curr)
        
        # Advance time
        t_current += sampling_time
        
    # 5. Plot Results
    print("\nPlotting results...")
    
    openad.plots.plot_mpc_results(
        history,
        d_max=0.5,
        title="MPC Maximizing Biogas",
        save_plot=True,
        show=True
    )

if __name__ == "__main__":
    run_control_example()
