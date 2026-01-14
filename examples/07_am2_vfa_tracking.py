"""
Example: AM2 MPC VFA Tracking
=============================

This script demonstrates how to use the Model Predictive Controller (MPC)
for the AM2 anaerobic digestion model to track a VFA (S2) setpoint.

Objective:
- Prevent VFA accumulation by tracking a specific setpoint.
- Maintain biogas production.
- Respect input constraints (Dilution rate).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from openad_lib.models.mechanistic import AM2Parameters
    from openad_lib.control import AM2MPC
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def run_vfa_tracking_example():
    print("=" * 60)
    print("AM2 MPC Control Demo: VFA Setpoint Tracking")
    print("=" * 60)
    
    # 1. Setup Parameters
    params = AM2Parameters()
    
    # 2. Initialize Controller
    print("\nInitializing MPC controller...")
    controller = AM2MPC(params)
    
    # Configuration
    sampling_time = 1.0  # days
    horizon = 10         # prediction horizon
    s2_setpoint = 2.0    # Target VFA concentration [g/L] (Low value to prevent accumulation)
    d_max = 0.5          # Max dilution rate
    
    print(f"Target VFA (S2) Setpoint: {s2_setpoint} g/L")
    print(f"Max Dilution Rate: {d_max} 1/d")
    
    controller.setup_controller(
        sampling_time=sampling_time,
        horizon=horizon,
        objective_type='tracking',
        tracking_variable='S2',
        setpoint=s2_setpoint,
        D_max=d_max
    )
    
    # Setup simulator for validation
    controller.setup_simulator(sampling_time=sampling_time)
    
    # 3. Initial Conditions
    # Start with high VFA to show regulation down to setpoint
    x0 = np.array([5.0, 1.0, 25.0, 1.0]) # [S1, X1, S2, X2]
    controller.set_initial_state(x0)
    
    # 4. Simulation Loop
    n_days = 50
    
    # Disturbance: Inlet substrate concentration
    S1in_nominal = 15.0
    pH_nominal = 7.0
    
    # Storage
    history = {
        'time': [],
        'D': [],
        'S2': [], 
        'Q': [],
        'Setpoint': []
    }
    
    print(f"\nStarting simulation for {n_days} days...")
    t_current = 0.0
    
    for k in range(n_days):
        # Disturbance
        S1in_k = S1in_nominal + 5.0 * np.sin(2 * np.pi * k / 20)
        pH_k = 7.0
        
        # Run MPC Step
        u_opt, y_next = controller.run_step(S1in_val=S1in_k, pH_val=pH_k)
        
        # Calculate Biogas (Q) for visualization
        # Extract states
        S2_curr = float(y_next[2])
        X2_curr = float(y_next[3])
        
        # Recalculate kinetics
        p = params
        mu2_base = p.m2 * (S2_curr / ((S2_curr**2)/p.Ki + S2_curr + p.K2))
        pH_factor = np.exp(-4 * ((pH_k - p.pHH)/(p.pHH - p.pHL))**2)
        mu2 = mu2_base * (1.0 - pH_factor)
        Q_curr = float(p.k6 * mu2 * X2_curr * p.c)
        
        # Store
        history['time'].append(t_current)
        history['D'].append(u_opt)
        history['S2'].append(S2_curr)
        history['Q'].append(Q_curr)
        history['Setpoint'].append(s2_setpoint)
        
        t_current += sampling_time
        
        print(f"Day {k}: D={u_opt:.3f}, S2={S2_curr:.2f} (Ref={s2_setpoint}), Q={Q_curr:.3f}")
        
    # 5. Plot Results
    print("\nPlotting results...")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Tracking Performance (S2)
    ax = axes[0]
    ax.plot(history['time'], history['S2'], 'b-', linewidth=2, label='VFA (S2)')
    ax.plot(history['time'], history['Setpoint'], 'k--', linewidth=2, label='Setpoint')
    ax.set_ylabel('Concentration [g/L]')
    ax.set_title(f'VFA Tracking Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Biogas Production
    ax = axes[1]
    ax.plot(history['time'], history['Q'], 'g-', label='Biogas Production')
    ax.set_ylabel('Rate [L/d]')
    ax.set_title('Biogas Production (Co-benefit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Control Input with Constraints
    ax = axes[2]
    ax.step(history['time'], history['D'], 'r-', where='post', label='Dilution Rate (D)')
    ax.axhline(y=d_max, color='r', linestyle=':', label='Max Constraint')
    ax.axhline(y=0, color='r', linestyle=':', label='Min Constraint')
    ax.set_ylabel('Rate [1/d]')
    ax.set_xlabel('Time [days]')
    ax.set_title('Control Input & Constraints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, d_max + 0.1)
    
    plt.tight_layout()
    plt.savefig('vfa_tracking_results.png')
    # plt.show()
    print("\nDone! Results saved to vfa_tracking_results.png")

if __name__ == "__main__":
    run_vfa_tracking_example()
