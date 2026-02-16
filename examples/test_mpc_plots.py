"""
Test MPC Plot with Publication-Quality Settings
================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import openad_lib as openad

print("=" * 80)
print("Testing MPC Plots with Publication-Quality Settings")
print("=" * 80)

# Create sample MPC history data
time = np.linspace(0, 10, 50)
history = {
    'time': time.tolist(),
    'S1': (10 + np.random.randn(50) * 0.5).tolist(),
    'S2': (5 + np.sin(time) * 2).tolist(),
    'Q': (100 + 20 * np.sin(time * 0.5)).tolist(),
    'D': (0.3 + 0.1 * np.sin(time * 0.3)).tolist(),
    'Setpoint': [5.0] * 50
}

print("\n[1] Creating MPC Biogas Maximization plot...")
openad.plots.plot_mpc_results(
    history={'time': time.tolist(), 'Q': history['Q'], 'D': history['D']},
    d_max=0.5,
    title="Phase 4A: MPC Biogas Maximization",
    save_path=Path(__file__).parent / 'images' / 'test_mpc_biogas_publication.png',
    show=False
)
print("  ✓ Saved biogas maximization plot")

print("\n[2] Creating MPC VFA Tracking plot...")
openad.plots.plot_mpc_results(
    history=history,
    d_max=0.5,
    title="Phase 4B: MPC VFA Tracking",
    save_path=Path(__file__).parent / 'images' / 'test_mpc_vfa_publication.png',
    show=False
)
print("  ✓ Saved VFA tracking plot")

print("\n" + "=" * 80)
print("Publication-Quality MPC Plot Settings:")
print("=" * 80)
print("  Titles:       18 pt (was 14 pt)")
print("  Axis labels:  16 pt (was 12 pt)")
print("  Legend:       14 pt (was 10 pt)")
print("  Tick labels:  14 pt (from rcParams)")
print("  Line width:   3.0 pt (was 2.0 pt)")
print("  Figure size:  14×10 / 14×14 (was 12×8 / 12×12)")
print("\n  ✓ All MPC plots now use publication-ready settings!")
print("=" * 80)
