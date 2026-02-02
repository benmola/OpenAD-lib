"""
Example: Probabilistic Feedstock with Uncertainty Quantification
=================================================================

This script demonstrates how to use OpenAD-lib's probabilistic feedstock
features to represent parameter uncertainty using probability distributions.

Features demonstrated:
1. Fitting distributions from multi-source measurement data
2. Generating ensemble realizations with physical constraints
3. Visualizing uncertainty in feedstock properties
4. Using probabilistic feedstocks in ADM1 simulations

Author: OpenAD-lib Team
Date: 2026
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import openad_lib as openad
from openad_lib.feedstock import (
    FeedstockDescriptor,
    BetaDistribution,
    LogNormalDistribution,
    assign_distribution
)

def run_probabilistic_feedstock_example():
    print("=" * 70)
    print("Probabilistic Feedstock Example: Uncertainty Quantification")
    print("=" * 70)
    
    # =================================================================
    # PART 1: Simulated Multi-Source Data for Maize Silage
    # =================================================================
    print("\n[1] Simulating multi-source measurement data...")
    print("    Sources: Laboratory (n=5), Literature (n=8), Expert (n=3)")
    
    # Simulate measurements from different sources
    # In practice, these would come from actual lab data, literature, etc.
    np.random.seed(42)
    
    # Total Solids [kg/m³] - from multiple facilities
    ts_lab = [310, 315, 308, 312, 318]
    ts_literature = [305, 320, 312, 308, 315, 310, 318, 313]
    ts_expert = [310, 315, 312]
    ts_all = ts_lab + ts_literature + ts_expert
    
    # Volatile Solids [g/kg TS]
    vs_lab = [945, 950, 948, 947, 952]
    vs_literature = [940, 955, 948, 945, 950, 947, 953, 949]
    vs_expert = [947, 950, 948]
    vs_all = vs_lab + vs_literature + vs_expert
    
    # BMP [NL CH4/kg VS]
    bmp_lab = [290, 295, 293, 292, 297]
    bmp_literature = [285, 300, 293, 290, 295, 292, 298, 294]
    bmp_expert = [293, 295, 292]
    bmp_all = bmp_lab + bmp_literature + bmp_expert
    
    # Total COD [kg COD/m³]
    cod_lab = [410, 425, 418, 415, 422]
    cod_literature = [405, 430, 420, 412, 425, 418, 428, 420]
    cod_expert = [418, 422, 420]
    cod_all = cod_lab + cod_literature + cod_expert
    
    # Proteins [kg/m³]
    proteins_all = [28, 32, 30, 31, 29, 33, 30, 31, 29, 32, 30, 31, 30, 31, 30, 31]
    
    print(f"    ✓ TS: {len(ts_all)} measurements")
    print(f"    ✓ VS: {len(vs_all)} measurements")
    print(f"    ✓ BMP: {len(bmp_all)} measurements")
    print(f"    ✓ COD: {len(cod_all)} measurements")
    print(f"    ✓ Proteins: {len(proteins_all)} measurements")
    
    # =================================================================
    # PART 2: Fit Probability Distributions (MLE)
    # =================================================================
    print("\n[2] Fitting probability distributions using MLE...")
    
    samples_dict = {
        'ts': ts_all,
        'vs': vs_all,
        'bmp': bmp_all,
        'cod_total': cod_all,
        'proteins': proteins_all
    }
    
    # Create feedstock with fitted distributions
    maize_probabilistic = FeedstockDescriptor.from_samples(
        name="Maize Silage (Probabilistic)",
        samples_dict=samples_dict
    )
    
    print(f"    ✓ Fitted distributions for {len(maize_probabilistic.uncertainty)} parameters")
    print("\n    Distribution Summary:")
    for param, dist in maize_probabilistic.uncertainty.items():
        print(f"      • {param:20s}: {dist}")
    
    # =================================================================
    # PART 3: Generate Ensemble Realizations
    # =================================================================
    print("\n[3] Generating ensemble realizations with physical constraints...")
    
    n_ensemble = 1000
    ensemble = maize_probabilistic.sample(
        n=n_ensemble,
        random_state=42,
        apply_constraints=True
    )
    
    print(f"    ✓ Generated {len(ensemble)} feedstock realizations")
    
    # Extract ensemble statistics
    ts_ensemble = [f.ts for f in ensemble]
    vs_ensemble = [f.vs for f in ensemble]
    bmp_ensemble = [f.bmp for f in ensemble]
    
    print(f"\n    Ensemble Statistics:")
    print(f"      TS:  mean={np.mean(ts_ensemble):.1f} ± {np.std(ts_ensemble):.1f} kg/m³")
    print(f"      VS:  mean={np.mean(vs_ensemble):.1f} ± {np.std(vs_ensemble):.1f} g/kg TS")
    print(f"      BMP: mean={np.mean(bmp_ensemble):.1f} ± {np.std(bmp_ensemble):.1f} NL CH4/kg VS")
    
    # Verify constraints
    print(f"\n    Physical Constraint Verification:")
    print(f"      ✓ All TS > 0: {all(f.ts > 0 for f in ensemble)}")
    print(f"      ✓ All VS ≤ 1000: {all(f.vs <= 1000 for f in ensemble)}")
    print(f"      ✓ All BMP > 0: {all(f.bmp > 0 for f in ensemble)}")
    
    # =================================================================
    # PART 4: Visualize Distributions
    # =================================================================
    print("\n[4] Visualizing probability distributions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Maize Silage: Probabilistic Feedstock Properties', fontsize=16, fontweight='bold')
    
    # TS Distribution
    ax = axes[0, 0]
    ax.hist(ts_all, bins=15, alpha=0.5, label='Measurements', color='steelblue', edgecolor='black')
    ax.hist(ts_ensemble, bins=50, alpha=0.6, label='Ensemble (n=1000)', color='orange', density=False)
    ax.axvline(np.mean(ts_all), color='blue', linestyle='--', linewidth=2, label='Mean (data)')
    ax.axvline(np.mean(ts_ensemble), color='red', linestyle='--', linewidth=2, label='Mean (ensemble)')
    ax.set_xlabel('Total Solids [kg/m³]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Total Solids Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # VS Distribution
    ax = axes[0, 1]
    ax.hist(vs_all, bins=15, alpha=0.5, label='Measurements', color='steelblue', edgecolor='black')
    ax.hist(vs_ensemble, bins=50, alpha=0.6, label='Ensemble (n=1000)', color='orange', density=False)
    ax.axvline(np.mean(vs_all), color='blue', linestyle='--', linewidth=2, label='Mean (data)')
    ax.axvline(np.mean(vs_ensemble), color='red', linestyle='--', linewidth=2, label='Mean (ensemble)')
    ax.set_xlabel('Volatile Solids [g/kg TS]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Volatile Solids Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # BMP Distribution
    ax = axes[1, 0]
    ax.hist(bmp_all, bins=15, alpha=0.5, label='Measurements', color='steelblue', edgecolor='black')
    ax.hist(bmp_ensemble, bins=50, alpha=0.6, label='Ensemble (n=1000)', color='orange', density=False)
    ax.axvline(np.mean(bmp_all), color='blue', linestyle='--', linewidth=2, label='Mean (data)')
    ax.axvline(np.mean(bmp_ensemble), color='red', linestyle='--', linewidth=2, label='Mean (ensemble)')
    ax.set_xlabel('BMP [NL CH4/kg VS]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Biochemical Methane Potential Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # COD Distribution
    ax = axes[1, 1]
    ax.hist(cod_all, bins=15, alpha=0.5, label='Measurements', color='steelblue', edgecolor='black')
    cod_ensemble = [f.cod_total for f in ensemble]
    ax.hist(cod_ensemble, bins=50, alpha=0.6, label='Ensemble (n=1000)', color='orange', density=False)
    ax.axvline(np.mean(cod_all), color='blue', linestyle='--', linewidth=2, label='Mean (data)')
    ax.axvline(np.mean(cod_ensemble), color='red', linestyle='--', linewidth=2, label='Mean (ensemble)')
    ax.set_xlabel('Total COD [kg COD/m³]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Total COD Distribution (LogNormal)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probabilistic_feedstock_distributions.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: probabilistic_feedstock_distributions.png")
    plt.show()
    
    # =================================================================
    # PART 5: Comparison with Deterministic Approach
    # =================================================================
    print("\n[5] Comparing with deterministic feedstock...")
    
    # Traditional deterministic feedstock (point estimates)
    maize_deterministic = FeedstockDescriptor(
        name="Maize Silage (Deterministic)",
        ts=np.mean(ts_all),
        vs=np.mean(vs_all),
        bmp=np.mean(bmp_all),
        proteins=np.mean(proteins_all)
    )
    
    print(f"\n    Deterministic (Point Estimates):")
    print(f"      TS:  {maize_deterministic.ts:.1f} kg/m³")
    print(f"      VS:  {maize_deterministic.vs:.1f} g/kg TS")
    print(f"      BMP: {maize_deterministic.bmp:.1f} NL CH4/kg VS")
    
    print(f"\n    Probabilistic (Mean ± Std):")
    print(f"      TS:  {np.mean(ts_ensemble):.1f} ± {np.std(ts_ensemble):.1f} kg/m³")
    print(f"      VS:  {np.mean(vs_ensemble):.1f} ± {np.std(vs_ensemble):.1f} g/kg TS")
    print(f"      BMP: {np.mean(bmp_ensemble):.1f} ± {np.std(bmp_ensemble):.1f} NL CH4/kg VS")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  ✓ Fitted Beta, LogNormal, and Gamma distributions from multi-source data")
    print("  ✓ Generated 1000-member ensemble with physical constraints")
    print("  ✓ Visualized uncertainty in feedstock properties")
    print("  ✓ Probabilistic approach captures natural variability")
    print("\nNext Steps:")
    print("  • Use ensemble in Monte Carlo ADM1 simulations")
    print("  • Propagate uncertainty through biogas predictions")
    print("  • Quantify prediction confidence intervals")
    print("=" * 70)

if __name__ == "__main__":
    run_probabilistic_feedstock_example()
