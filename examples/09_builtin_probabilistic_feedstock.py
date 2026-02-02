"""
Example: Using Built-in Probabilistic Feedstock Library
========================================================

This script demonstrates how to use the built-in uncertainty data
in the FeedstockLibrary to generate probabilistic feedstocks without
providing your own measurement data.

Author: OpenAD-lib Team
Date: 2026
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from openad_lib.feedstock import FeedstockLibrary

def run_builtin_probabilistic_example():
    print("=" * 70)
    print("Built-in Probabilistic Feedstock Library Example")
    print("=" * 70)
    
    # Initialize library
    lib = FeedstockLibrary()
    
    print("\n[1] Available feedstocks in library:")
    for i, name in enumerate(lib.list_feedstocks(), 1):
        print(f"    {i:2d}. {name}")
    
    # =================================================================
    # OPTION 1: Get deterministic feedstock (traditional approach)
    # =================================================================
    print("\n[2] Deterministic feedstock (point estimates):")
    maize_det = lib.get("Maize")
    print(f"    Maize (Deterministic):")
    print(f"      TS:  {maize_det.ts:.1f} kg/m³")
    print(f"      VS:  {maize_det.vs:.1f} g/kg TS")
    print(f"      BMP: {maize_det.bmp:.1f} NL CH4/kg VS")
    print(f"      Has uncertainty: {maize_det.uncertainty is not None}")
    
    # =================================================================
    # OPTION 2: Get probabilistic feedstock (NEW!)
    # =================================================================
    print("\n[3] Probabilistic feedstock with USER DATA + LIBRARY:")
    
    # Realistic scenario: User has LIMITED measurements (sparse data)
    # They only measured TS and BMP, missing VS, COD, proteins, lipids
    user_measurements = {
        'ts': [310, 315, 308],      # Only 3 TS measurements from their lab
        'bmp': [290, 295]           # Only 2 BMP measurements
        # Missing: vs, cod_total, proteins, lipids
    }
    
    print(f"    User's measurements:")
    print(f"      TS:  {user_measurements['ts']} (n={len(user_measurements['ts'])})")
    print(f"      BMP: {user_measurements['bmp']} (n={len(user_measurements['bmp'])})")
    print(f"      Missing: VS, COD, proteins, lipids")
    print(f"\n    → Library will fill in missing parameters!")
    
    # Get probabilistic feedstock: merges user data + library uncertainty
    maize_prob = lib.get_probabilistic("Maize", user_data=user_measurements)
    
    print(f"\n    {maize_prob.name}:")
    print(f"      TS:  {maize_prob.ts:.1f} kg/m³ (from USER data)")
    print(f"      VS:  {maize_prob.vs:.1f} g/kg TS (from LIBRARY)")
    print(f"      BMP: {maize_prob.bmp:.1f} NL CH4/kg VS (from USER data)")
    print(f"      COD: {maize_prob.cod_total:.1f} kg COD/m³ (from LIBRARY)")
    print(f"      Has uncertainty: {maize_prob.uncertainty is not None}")
    print(f"\n    Fitted distributions:")
    for param, dist in maize_prob.uncertainty.items():
        source = "USER" if param in user_measurements else "LIBRARY"
        print(f"      • {param:12s}: {dist} [{source}]")
    
    # =================================================================
    # OPTION 3: Generate ensemble directly (convenience method)
    # =================================================================
    print("\n[4] Generating ensemble (1000 realizations)...")
    ensemble = lib.generate_ensemble("Maize", n_realizations=1000, random_state=42)
    
    # Extract statistics
    ts_values = [f.ts for f in ensemble]
    vs_values = [f.vs for f in ensemble]
    bmp_values = [f.bmp for f in ensemble]
    
    print(f"    ✓ Generated {len(ensemble)} realizations")
    print(f"\n    Ensemble Statistics:")
    print(f"      TS:  {np.mean(ts_values):.1f} ± {np.std(ts_values):.1f} kg/m³")
    print(f"      VS:  {np.mean(vs_values):.1f} ± {np.std(vs_values):.1f} g/kg TS")
    print(f"      BMP: {np.mean(bmp_values):.1f} ± {np.std(bmp_values):.1f} NL CH4/kg VS")
    
    # Verify constraints
    print(f"\n    Constraint Verification:")
    print(f"      ✓ All TS > 0: {all(f.ts > 0 for f in ensemble)}")
    print(f"      ✓ All VS ≤ 1000: {all(f.vs <= 1000 for f in ensemble)}")
    print(f"      ✓ All BMP > 0: {all(f.bmp > 0 for f in ensemble)}")
    
    # =================================================================
    # VISUALIZATION: Compare user's sparse measurements with ensemble
    # =================================================================
    print("\n[4.5] Visualizing probability distributions...")
    print("    Comparing user's LIMITED measurements with full ensemble...")
    
    # User's actual sparse measurements (what they provided)
    user_ts = np.array(user_measurements['ts'])  # Only 3 measurements
    user_bmp = np.array(user_measurements['bmp'])  # Only 2 measurements
    
    # For VS and COD, user had NO measurements (library filled these)
    print(f"\n    User provided:")
    print(f"      TS:  {len(user_ts)} measurements")
    print(f"      BMP: {len(user_bmp)} measurements")
    print(f"      VS:  0 measurements (library filled)")
    print(f"      COD: 0 measurements (library filled)")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Maize Silage: Sparse User Data + Library Uncertainty', 
                 fontsize=16, fontweight='bold')
    
    # TS Distribution - USER PROVIDED
    ax = axes[0, 0]
    ax.hist(user_ts, bins=3, alpha=0.6, label=f'User Measurements (n={len(user_ts)})', 
            color='darkgreen', edgecolor='black', width=2)
    ax.hist(ts_values, bins=50, alpha=0.5, label='Full Ensemble (n=1000)', 
            color='lightblue', density=False)
    ax.axvline(np.mean(user_ts), color='darkgreen', linestyle='--', linewidth=2.5, 
               label='Mean (user)')
    ax.axvline(np.mean(ts_values), color='blue', linestyle='--', linewidth=2, 
               label='Mean (ensemble)')
    ax.set_xlabel('Total Solids [kg/m³]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('TS: User Provided (3 measurements)', fontsize=12, color='darkgreen')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # VS Distribution - LIBRARY FILLED
    ax = axes[0, 1]
    ax.hist(vs_values, bins=50, alpha=0.7, label='Library Ensemble (n=1000)', 
            color='orange', edgecolor='black')
    ax.axvline(np.mean(vs_values), color='red', linestyle='--', linewidth=2, 
               label='Mean (library)')
    ax.text(0.05, 0.95, 'No user data\n(Library filled)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Volatile Solids [g/kg TS]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('VS: Library Filled (user had no data)', fontsize=12, color='orange')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # BMP Distribution - USER PROVIDED
    ax = axes[1, 0]
    ax.hist(user_bmp, bins=2, alpha=0.6, label=f'User Measurements (n={len(user_bmp)})', 
            color='darkgreen', edgecolor='black', width=2)
    ax.hist(bmp_values, bins=50, alpha=0.5, label='Full Ensemble (n=1000)', 
            color='lightblue', density=False)
    ax.axvline(np.mean(user_bmp), color='darkgreen', linestyle='--', linewidth=2.5, 
               label='Mean (user)')
    ax.axvline(np.mean(bmp_values), color='blue', linestyle='--', linewidth=2, 
               label='Mean (ensemble)')
    ax.set_xlabel('BMP [NL CH4/kg VS]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('BMP: User Provided (2 measurements)', fontsize=12, color='darkgreen')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # COD Distribution - LIBRARY FILLED
    ax = axes[1, 1]
    cod_values = [f.cod_total for f in ensemble]
    ax.hist(cod_values, bins=50, alpha=0.7, label='Library Ensemble (n=1000)', 
            color='orange', edgecolor='black')
    ax.axvline(np.mean(cod_values), color='red', linestyle='--', linewidth=2, 
               label='Mean (library)')
    ax.text(0.05, 0.95, 'No user data\n(Library filled)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Total COD [kg COD/m³]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('COD: Library Filled (user had no data)', fontsize=12, color='orange')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('builtin_probabilistic_feedstock_distributions.png', dpi=300, bbox_inches='tight')
    print("\n    ✓ Saved: builtin_probabilistic_feedstock_distributions.png")
    print("\n    Interpretation:")
    print("      • Dark green: User's sparse measurements (TS: 3, BMP: 2)")
    print("      • Light blue: Ensemble from user's data (where provided)")
    print("      • Orange: Library filled missing parameters (VS, COD)")
    print("      → Library completes incomplete datasets!")
    plt.show()
    
    # =================================================================
    # OPTION 4: Try different feedstocks
    # =================================================================
    print("\n[5] Comparing uncertainty across feedstocks:")
    feedstocks_to_compare = ["Maize", "Grass", "Chicken Litter"]
    
    for feedstock_name in feedstocks_to_compare:
        ensemble = lib.generate_ensemble(feedstock_name, n_realizations=1000, random_state=42)
        ts_values = [f.ts for f in ensemble]
        bmp_values = [f.bmp for f in ensemble]
        
        cv_ts = (np.std(ts_values) / np.mean(ts_values)) * 100  # Coefficient of variation
        cv_bmp = (np.std(bmp_values) / np.mean(bmp_values)) * 100
        
        print(f"\n    {feedstock_name}:")
        print(f"      TS:  {np.mean(ts_values):6.1f} ± {np.std(ts_values):4.1f} kg/m³ (CV: {cv_ts:.1f}%)")
        print(f"      BMP: {np.mean(bmp_values):6.1f} ± {np.std(bmp_values):4.1f} NL CH4/kg VS (CV: {cv_bmp:.1f}%)")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Built-in uncertainty data for all 12 feedstocks")
    print("  ✓ get_probabilistic() - Get feedstock with fitted distributions")
    print("  ✓ generate_ensemble() - Generate ensemble in one call")
    print("  ✓ No need to provide your own measurement data")
    print("\nUsage:")
    print("  # Simple one-liner to get 1000 realizations:")
    print("  ensemble = lib.generate_ensemble('Maize', n_realizations=1000)")
    print("=" * 70)

if __name__ == "__main__":
    run_builtin_probabilistic_example()
