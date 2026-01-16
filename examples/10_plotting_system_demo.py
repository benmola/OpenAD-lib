"""
Example: Using OpenAD Unified Plotting System
==============================================

This script demonstrates the new unified plotting API that provides
consistent, professional plots across all models.

Features demonstrated:
1. Single output predictions with confidence intervals
2. Multi-output predictions (MTGP style)
3. Training curves
4. Residual analysis
5. Automatic plot type selection

All plots match the professional OpenAD style!
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import openad_lib as openad

def demo_simple_prediction():
    """Demo 1: Simple prediction plot"""
    print("="*70)
    print("DEMO 1: Simple Prediction Plot")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 150, 150)
    y_true = 2 * np.sin(x/20) + 5 + np.random.normal(0, 0.3, 150)
    y_pred = 2 * np.sin(x/20) + 5 + np.random.normal(0, 0.15, 150)
    
    # Simple plot
    openad.plots.plot_predictions(
        y_true, y_pred, x=x,
        title="LSTM Biogas Prediction",
        xlabel="Time (days)",
        ylabel="Biogas (m³/day)",
        save_path=current_dir.parent / "images" / "demo_simple_plot.png"
    )
    print("✓ Simple prediction plot created\n")


def demo_train_test_split():
    """Demo 2: Plot with train/test split"""
    print("="*70)
    print("DEMO 2: Train/Test Split Visualization")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 150
    x = np.arange(n_samples)
    y_true = 2 * np.sin(x/20) + 5 + np.random.normal(0, 0.3, n_samples)
    y_pred = 2 * np.sin(x/20) + 5 + np.random.normal(0, 0.15, n_samples)
    
    # Split indices
    split_idx = 120
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, n_samples)
    
    # Plot with train/test distinction
    openad.plots.plot_predictions(
        y_true, y_pred, x=x,
        train_indices=train_idx,
        test_indices=test_idx,
        title="Model Performance (Train vs Test)",
        xlabel="Time (days)",
        ylabel="VFA (g COD/L)",
        save_path=current_dir.parent / "images" / "demo_train_test_plot.png"
    )
    print("✓ Train/test split plot created\n")


def demo_confidence_intervals():
    """Demo 3: Predictions with uncertainty"""
    print("="*70)
    print("DEMO 3: Predictions with Confidence Intervals")
    print("="*70)
    
    # Generate sample data with uncertainty
    np.random.seed(42)
    x = np.linspace(0, 150, 150)
    y_true = 2 * np.sin(x/20) + 5 + np.random.normal(0, 0.3, 150)
    y_pred = 2 * np.sin(x/20) + 5
    
    # Generate confidence bounds
    uncertainty = 0.5 + 0.3 * np.abs(np.sin(x/30))  # Varying uncertainty
    y_lower = y_pred - 1.96 * uncertainty
    y_upper = y_pred + 1.96 * uncertainty
    
    # Plot with confidence intervals
    openad.plots.plot_predictions(
        y_true, y_pred, x=x,
        y_lower=y_lower, y_upper=y_upper,
        title="GP Prediction with Uncertainty",
        xlabel="Time",
        ylabel="SCOD (g COD/L)",
        save_path=current_dir.parent / "images" / "demo_confidence_plot.png"
    )
    print("✓ Confidence interval plot created\n")


def demo_multi_output():
    """Demo 4: Multi-output predictions (like MTGP)"""
    print("="*70)
    print("DEMO 4: Multi-Output Predictions (MTGP Style)")
    print("="*70)
    
    # Generate multi-output data
    np.random.seed(42)
    n_samples = 150
    x = np.arange(n_samples)
    
    # Three outputs with different dynamics
    y_true = np.column_stack([
        5 + 2*np.sin(x/20) + np.random.normal(0, 0.5, n_samples),    # SCODout
        15 + 5*np.sin(x/15) + np.random.normal(0, 1.0, n_samples),   # VFAout
        2 + 0.5*np.sin(x/25) + np.random.normal(0, 0.2, n_samples)   # Biogas
    ])
    
    y_pred = np.column_stack([
        5 + 2*np.sin(x/20) + np.random.normal(0, 0.2, n_samples),
        15 + 5*np.sin(x/15) + np.random.normal(0, 0.5, n_samples),
        2 + 0.5*np.sin(x/25) + np.random.normal(0, 0.1, n_samples)
    ])
    
    # Confidence bounds for each output
    y_lower = y_pred - np.column_stack([
        np.ones(n_samples) * 0.8,
        np.ones(n_samples) * 2.0,
        np.ones(n_samples) * 0.3
    ])
    y_upper = y_pred + np.column_stack([
        np.ones(n_samples) * 0.8,
        np.ones(n_samples) * 2.0,
        np.ones(n_samples) * 0.3
    ])
    
    # Train/test split
    split_idx = 120
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, n_samples)
    
    # Plot multi-output (matches your example figure!)
    openad.plots.plot_multi_output(
        y_true, y_pred, x=x,
        y_lower=y_lower, y_upper=y_upper,
        train_indices=train_idx,
        test_indices=test_idx,
        output_names=['SCODout', 'VFAout', 'Biogas'],
        xlabel="Time",
        save_path=current_dir.parent / "images" / "demo_multi_output_plot.png"
    )
    print("✓ Multi-output plot created (matches your example!)\n")


def demo_training_curves():
    """Demo 5: Training history"""
    print("="*70)
    print("DEMO 5: Training Curves")
    print("="*70)
    
    # Simulate training history
    epochs = 50
    train_losses = 3.0 * np.exp(-np.arange(epochs) / 10) + np.random.normal(0, 0.1, epochs)
    val_losses = 3.2 * np.exp(-np.arange(epochs) / 10) + np.random.normal(0, 0.15, epochs)
    
    openad.plots.plot_training_curves(
        train_losses, val_losses,
        title="LSTM Training History",
        save_path=current_dir.parent / "images" / "demo_training_curves.png"
    )
    print("✓ Training curves plot created\n")


def demo_residuals():
    """Demo 6: Residual analysis"""
    print("="*70)
    print("DEMO 6: Residual Analysis")
    print("="*70)
    
    # Generate data with some bias
    np.random.seed(42)
    y_true = np.random.normal(5, 2, 200)
    y_pred = y_true + np.random.normal(0.2, 0.5, 200)  # Slight bias
    
    openad.plots.plot_residuals(
        y_true, y_pred,
        title="Model Residual Analysis",
        save_path=current_dir.parent / "images" / "demo_residuals_plot.png"
    )
    print("✓ Residual plot created\n")


def demo_quick_plot():
    """Demo 7: Automatic plot type selection"""
    print("="*70)
    print("DEMO 7: Quick Plot (Automatic)")
    print("="*70)
    
    # 1D data - automatically uses plot_predictions
    y_true_1d = np.random.normal(5, 1, 100) 
    y_pred_1d = y_true_1d + np.random.normal(0, 0.3, 100)
    
    print("Using quick_plot with 1D data...")
    openad.plots.quick_plot(
        y_true_1d, y_pred_1d,
        title="Auto-detected: Single Output"
    )
    
    # 2D data - automatically uses plot_multi_output
    y_true_2d = np.random.normal(5, 1, (100, 3))
    y_pred_2d = y_true_2d + np.random.normal(0, 0.3, (100, 3))
    
    print("Using quick_plot with 2D data...")
    openad.plots.quick_plot(
        y_true_2d, y_pred_2d,
        output_names=['Output 1', 'Output 2', 'Output 3'],
        title="Auto-detected: Multi-Output",
        save_path=current_dir.parent / "images" / "demo_quick_plot.png"
    )
    print("✓ Quick plots created\n")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("OPENAD UNIFIED PLOTTING SYSTEM DEMO")
    print("="*70 + "\n")
    
    # Create images directory
    images_dir = current_dir.parent / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Run all demos
    demo_simple_prediction()
    demo_train_test_split()
    demo_confidence_intervals()
    demo_multi_output()  # This matches your example figure!
    demo_training_curves()
    demo_residuals()
    demo_quick_plot()
    
    print("="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print(f"\nPlots saved to: {images_dir}")
    print("\nKey features:")
    print("  ✓ Consistent professional style")
    print("  ✓ Automatic handling of train/test splits")
    print("  ✓ Confidence intervals support")
    print("  ✓ Multi-output plotting (MTGP style)")
    print("  ✓ No matplotlib knowledge needed!")
    print("\nUsage:")
    print("  import openad_lib as openad")
    print("  openad.plots.plot_predictions(y_true, y_pred)")
    print("  openad.plots.plot_multi_output(Y_true, Y_pred)")


if __name__ == "__main__":
    main()
