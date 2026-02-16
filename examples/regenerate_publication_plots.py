"""
Regenerate Case Study Plots with Publication-Quality Settings
==============================================================

This script regenerates the case study plots with the new publication-quality
font sizes and line widths for better readability in papers.
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openad_lib as openad

print("=" * 80)
print("Regenerating Case Study Plots with Publication-Quality Settings")
print("=" * 80)

# Apply publication-quality style
openad.plots.set_openad_style()

# =============================================================================
# Example: Phase 1 - ADM1 Baseline (Simplified for demonstration)
# =============================================================================
print("\n[1] Creating sample ADM1-style plot...")

# Create sample data (simulating ADM1 results)
np.random.seed(42)
time = np.linspace(0, 150, 150)
actual = 38000 + np.random.normal(0, 500, 150)
predicted = np.zeros_like(time)

# Simulate startup and steady state
for i in range(len(time)):
    if time[i] < 20:
        predicted[i] = -2000 + time[i] * 2000
    elif time[i] < 60:
        predicted[i] = 20000 + (time[i] - 20) * 500
    else:
        predicted[i] = 40000 + np.sin((time[i] - 60) / 10) * 2000

# Create publication-quality plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data with publication settings
ax.scatter(time, actual, alpha=0.6, s=50, label='Actual', color='#4472C4')
ax.plot(time, predicted, linewidth=3, label='Predicted', color='#000000')

# Labels and title with large fonts
ax.set_xlabel('Time (days)', fontweight='bold')
ax.set_ylabel('Biogas Production (m³/day)', fontweight='bold')
ax.set_title('Phase 1: ADM1 Mechanistic Baseline', fontweight='bold', pad=20)

# Legend with larger font
ax.legend(loc='upper left', frameon=True)

# Grid
ax.grid(True, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save with high DPI for publication
output_path = current_dir / 'images' / 'phase1_adm1_baseline_publication.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Plot Settings Applied:")
print("=" * 80)
print("  Font sizes:")
print("    - Title: 18 pt (was 12 pt)")
print("    - Axis labels: 16 pt (was 10 pt)")
print("    - Tick labels: 14 pt (was 9 pt)")
print("    - Legend: 14 pt (was 9 pt)")
print("\n  Line properties:")
print("    - Line width: 3.0 pt (was 2.0 pt)")
print("    - Marker size: 8 pt (was 6 pt)")
print("\n  ✓ All plots will now use these publication-ready settings!")
print("=" * 80)

print("\nTo regenerate all case study plots with new settings:")
print("  Run: python examples/case_study_integrated.py")
print("\n" + "=" * 80)
