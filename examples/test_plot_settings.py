"""Test publication-quality plot settings"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import openad_lib as openad
from matplotlib import rcParams

# Apply OpenAD style
openad.plots.set_openad_style()

print("=" * 60)
print("Publication-Quality Plot Settings")
print("=" * 60)
print("\nFont Sizes:")
print(f"  Base font:        {rcParams['font.size']} pt")
print(f"  Title:            {rcParams['axes.titlesize']} pt")
print(f"  Axis labels:      {rcParams['axes.labelsize']} pt")
print(f"  Tick labels:      {rcParams['xtick.labelsize']} pt")
print(f"  Legend:           {rcParams['legend.fontsize']} pt")

print("\nLine Properties:")
print(f"  Line width:       {rcParams['lines.linewidth']} pt")
print(f"  Marker size:      {rcParams['lines.markersize']} pt")
print(f"  Axes line width:  {rcParams['axes.linewidth']} pt")

print("\n✓ Settings updated for publication-quality figures!")
print("=" * 60)
