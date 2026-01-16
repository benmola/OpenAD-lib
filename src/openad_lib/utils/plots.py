"""
Unified plotting system for openad_lib.

Provides consistent, publication-quality plots for all models without
requiring users to know matplotlib.

Features:
- Consistent style across all model types
- Automatic plot generation from model results
- Professional appearance with confidence intervals
- Easy-to-use API: openad.plots.plot_predictions(...)

Example:
    >>> import openad_lib as openad
    >>> openad.plots.plot_predictions(y_true, y_pred, title="Model Results")
    >>> openad.plots.plot_multi_output(results, save_path="results.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional, List, Tuple, Union, Dict
from pathlib import Path


# ============================================================================
# OPENAD PLOTTING STYLE CONFIGURATION
# ============================================================================

def set_openad_style():
    """
    Set consistent OpenAD plotting style across all figures.
    
    This style matches the professional appearance shown in examples:
    - Clean white background
    - Light grid
    - Optimized for both screen and print
    - Consistent colors and markers
    """
    plt.style.use('default')  # Reset to ensure consistency
    
    rcParams['figure.facecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = '#333333'
    rcParams['axes.linewidth'] = 1.0
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linestyle'] = '-'
    rcParams['grid.linewidth'] = 0.8
    rcParams['grid.color'] = '#D3D3D3'
    
    # Font settings
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['font.size'] = 10
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    
    # Line and marker settings
    rcParams['lines.linewidth'] = 2.0
    rcParams['lines.markersize'] = 6
    
    # Legend
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.edgecolor'] = '#CCCCCC'
    rcParams['legend.fancybox'] = False


# Default colors matching the example figure
COLORS = {
    'train': '#4472C4',      # Blue for training data
    'test': '#E74C3C',       # Red for test data  
    'predicted': '#000000',  # Black for predictions
    'confidence': '#A9A9A9', # Gray for confidence intervals
    'actual': '#2E86DE',     # Blue for actual values
}


def get_images_dir() -> Path:
    """
    Get the default images directory for saving plots.
    
    Tries to find the project's images folder, or creates one in the current
    working directory if not found.
    
    Returns
    -------
    images_dir : Path
        Path to images directory (created if doesn't exist)
        
    Example
    -------
    >>> import openad_lib as openad
    >>> images_dir = openad.plots.get_images_dir()
    >>> print(images_dir)
    /path/to/project/images
    """
    from openad_lib.config import config
    
    # Try to use project root from config
    try:
        # data_dir is src/openad_lib/data
        # .parent -> src/openad_lib
        # .parent.parent -> src
        # .parent.parent.parent -> project_root (OpenAD-lib)
        project_root = config.data_dir.parent.parent.parent
        images_dir = project_root / 'images'
    except Exception:
        # Fallback to current working directory
        images_dir = Path.cwd() / 'images'
    
    # Create directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir



def resolve_save_path(
    save_path: Optional[Union[str, Path]], 
    save_plot: bool, 
    default_filename: str
) -> Optional[Path]:
    """
    Resolve the final save path based on user arguments.
    
    Logic:
    1. If save_path is provided, use it.
    2. If save_plot is True, use default directory + default_filename.
    3. Otherwise, return None (don't save).
    """
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    if save_plot:
        return get_images_dir() / default_filename
        
    return None


# ============================================================================
# CORE PLOTTING FUNCTIONS
# ============================================================================

def plot_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    x: Optional[Union[np.ndarray, pd.Series]] = None,
    y_lower: Optional[Union[np.ndarray, pd.Series]] = None,
    y_upper: Optional[Union[np.ndarray, pd.Series]] = None,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    title: str = "Predictions vs Actual",
    xlabel: str = "Time",
    ylabel: str = "Value",
    save_path: Optional[Union[str, Path]] = None,
    save_plot: bool = False,
    show: bool = True
) -> plt.Figure:
    """
    Plot predictions vs actual values with optional confidence intervals.
    
    Parameters
    ----------
    ...
    save_path : str or Path, optional
        Specific path to save figure. Overrides save_plot.
    save_plot : bool, default=False
        If True, auto-saves to default images directory using generated filename.
    ...
    """
    set_openad_style()
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if x is None:
        x = np.arange(len(y_true))
    else:
        x = np.asarray(x).flatten()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot train/test split if provided
    if train_indices is not None and test_indices is not None:
        ax.scatter(x[train_indices], y_true[train_indices], 
                  c=COLORS['train'], marker='o', s=40, alpha=0.6, 
                  label='True (Train)', zorder=3)
        ax.scatter(x[test_indices], y_true[test_indices],
                  c=COLORS['test'], marker='o', s=40, alpha=0.7,
                  label='True (Test)', zorder=3)
    else:
        # Plot all actual values
        ax.scatter(x, y_true, c=COLORS['actual'], marker='o', s=40, 
                  alpha=0.6, label='Actual', zorder=3)
    
    # Plot predictions
    ax.plot(x, y_pred, color=COLORS['predicted'], linewidth=2.5,
           label='Predicted', zorder=4)
    
    # Plot confidence interval if provided
    if y_lower is not None and y_upper is not None:
        y_lower = np.asarray(y_lower).flatten()
        y_upper = np.asarray(y_upper).flatten()
        ax.fill_between(x, y_lower, y_upper, 
                        color=COLORS['confidence'], alpha=0.3,
                        label='95% Confidence', zorder=2)
    
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Handle saving
    default_name = f"{title.lower().replace(' ', '_')}.png"
    final_path = resolve_save_path(save_path, save_plot, default_name)
    
    if final_path:
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {final_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_multi_output(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    output_names: Optional[List[str]] = None,
    title: str = "Multi-Output Prediction",
    xlabel: str = "Time",
    save_path: Optional[Union[str, Path]] = None,
    save_plot: bool = False,
    show: bool = True
) -> plt.Figure:
    """
    Plot multiple outputs in subplots.
    
    Parameters
    ----------
    ...
    save_path : str or Path, optional
        Specific path to save figure. Overrides save_plot.
    save_plot : bool, default=False
        If True, auto-saves to default images directory using generated filename.
    ...
    """
    set_openad_style()
    
    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    n_samples, n_outputs = y_true.shape
    
    if x is None:
        x = np.arange(n_samples)
    
    # Sort all data by x-axis for smooth lines
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    
    if y_lower is not None:
        y_lower = y_lower[sort_idx]
    if y_upper is not None:
        y_upper = y_upper[sort_idx]
    
    # Adjust train/test indices after sorting
    if train_indices is not None:
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[train_indices] = True
        train_mask = train_mask[sort_idx]
        train_indices = np.where(train_mask)[0]
    
    if test_indices is not None:
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_indices] = True
        test_mask = test_mask[sort_idx]
        test_indices = np.where(test_mask)[0]
    
    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]
    
    # Create subplots
    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 4 * n_outputs))
    if n_outputs == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot confidence interval first (background)
        if y_lower is not None and y_upper is not None:
            ax.fill_between(x, y_lower[:, i], y_upper[:, i],
                           color=COLORS['confidence'], alpha=0.3,
                           label='95% Confidence', zorder=2)
        
        # Plot train/test split if provided
        if train_indices is not None and test_indices is not None:
            ax.scatter(x[train_indices], y_true[train_indices, i],
                      c=COLORS['train'], marker='o', s=40, alpha=0.6,
                      label=f'True {output_names[i]} (Train)', zorder=3)
            ax.scatter(x[test_indices], y_true[test_indices, i],
                      c=COLORS['test'], marker='o', s=40, alpha=0.7,
                      label=f'True {output_names[i]} (Test)', zorder=3)
        else:
            ax.scatter(x, y_true[:, i], c=COLORS['actual'], marker='o',
                      s=40, alpha=0.6, label=f'True {output_names[i]}', zorder=3)
        
        # Plot predictions (smooth line on top)
        ax.plot(x, y_pred[:, i], color=COLORS['predicted'], linewidth=2.5,
               label=f'Predicted {output_names[i]}', zorder=4)
        
        ax.set_title(output_names[i], fontweight='bold')
        ax.set_ylabel(output_names[i])
        
        # Only show xlabel on bottom plot
        if i == n_outputs - 1:
            ax.set_xlabel(xlabel)
        
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    
    # Handle saving
    default_name = f"{title.lower().replace(' ', '_')}.png"
    final_path = resolve_save_path(save_path, save_plot, default_name)
    
    if final_path:
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {final_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training History",
    save_path: Optional[Union[str, Path]] = None,
    save_plot: bool = False,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Parameters
    ----------
    ...
    save_plot : bool, default=False
    ...
    """
    set_openad_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=COLORS['train'], linewidth=2,
           label='Training Loss', marker='o', markersize=4)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, color=COLORS['test'], linewidth=2,
               label='Validation Loss', marker='s', markersize=4)
    
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Handle saving
    default_name = f"{title.lower().replace(' ', '_')}.png"
    final_path = resolve_save_path(save_path, save_plot, default_name)
    
    if final_path:
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {final_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: Optional[Union[str, Path]] = None,
    save_plot: bool = False,
    show: bool = True
) -> plt.Figure:
    """
    Plot residuals (prediction errors) analysis.
    
    Parameters
    ----------
    ...
    save_plot : bool, default=False
    ...
    """
    set_openad_style()
    
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, c=COLORS['actual'], alpha=0.6, s=40)
    ax1.axhline(y=0, color=COLORS['predicted'], linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, color=COLORS['actual'], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Handle saving
    default_name = f"{title.lower().replace(' ', '_')}.png"
    final_path = resolve_save_path(save_path, save_plot, default_name)
    
    if final_path:
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {final_path}")
    
    if show:
        plt.show()
    
    return fig

def plot_calibration_comparison(
    initial_results: pd.DataFrame,
    final_results: pd.DataFrame,
    variables: List[str] = ['S1', 'S2', 'Q'],
    labels: Optional[List[str]] = None,
    title: str = "Calibration Comparison",
    save_path: Optional[Union[str, Path]] = None,
    save_plot: bool = False,
    show: bool = True
) -> plt.Figure:
    """
    Plot calibration comparison results.

    Parameters
    ----------
    initial_results : pd.DataFrame
        Results from the initial simulation (before calibration).
    final_results : pd.DataFrame
        Results from the final simulation (after calibration), including measured data if available.
    variables : List[str], default=['S1', 'S2', 'Q']
        List of variable names to plot.
    labels : List[str], optional
        List of display labels for the variables. If None, uses variable names.
    title : str, default="Calibration Comparison"
        Overall title for the plot.
    save_path : str or Path, optional
        Specific path to save figure.
    save_plot : bool, default=False
        If True, auto-saves to default images directory.
    show : bool, default=True
        Whether to show the plot.
    """
    # Use BMH style as requested, but ensure other OpenAD defaults are respected where possible
    plt.style.use('bmh')
    
    if labels is None:
        labels = variables
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 10), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
        
    time = final_results['time']
    
    for i, var in enumerate(variables):
        ax = axes[i]
        
        # Measured Data
        if f'{var}_measured' in final_results.columns:
            valid = ~final_results[f'{var}_measured'].isna()
            ax.plot(time[valid], final_results[f'{var}_measured'][valid], 
                    'o', color='#2E86C1', markersize=6, label='Measured', alpha=0.7)
            
        # Initial Model
        if var in initial_results.columns:
             ax.plot(time, initial_results[var], '--', color='gray', linewidth=2, 
                    label='Initial Model', alpha=0.7)
        
        # Calibrated Model
        if var in final_results.columns:
            ax.plot(time, final_results[var], '-', color='#27AE60', linewidth=2, 
                    label='Calibrated Model')
        
        ax.set_ylabel(labels[i], fontsize=14, fontweight='bold')
        ax.set_title(f'{labels[i]} Calibration Comparison', fontsize=16, pad=20)
        ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    axes[-1].set_xlabel('Time (days)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Handle saving
    default_name = f"{title.lower().replace(' ', '_')}.png"
    final_path = resolve_save_path(save_path, save_plot, default_name)
    
    if final_path:
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {final_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_plot(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    **kwargs
):
    """
    Intelligently plot based on data dimensions.
    
    Automatically chooses between plot_predictions() and plot_multi_output()
    based on whether data is 1D or 2D.
    
    Parameters
    ----------
    y_true : array-like or DataFrame
        True values
    y_pred : array-like or DataFrame
        Predicted values
    **kwargs
        Additional arguments passed to plotting function
        
    Examples
    --------
    >>> import openad_lib as openad
    >>> # Automatically handles 1D or multi-output
    >>> openad.plots.quick_plot(y_true, y_pred, title="Results")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        # Single output
        return plot_predictions(y_true, y_pred, **kwargs)
    else:
        # Multi-output
        return plot_multi_output(y_true, y_pred, **kwargs)



def plot_mpc_results(
    history: Dict[str, List[float]],
    d_max: Optional[float] = None,
    s2_setpoint: Optional[float] = None,
    title: str = "MPC Control Results",
    save_path: Optional[Union[str, Path]] = None,
    save_plot: bool = False,
    show: bool = True
) -> plt.Figure:
    """
    Plot MPC control results (States, Biogas, Control Input).

    Parameters
    ----------
    history : Dict
        Dictionary containing 'time', 'S1', 'S2', 'Q', 'D' lists.
        Can optionally contain 'Setpoint'.
    d_max : float, optional
        Maximum dilution rate constraint to display.
    s2_setpoint : float, optional
        S2 setpoint to display if not in history.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save the plot.
    save_plot : bool
        Whether to auto-save.
    show : bool
        Whether to show the plot.
    """
    set_openad_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Determine what to plot in the first subplot (Tracking vs State)
    ax = axes[0]
    if 'Setpoint' in history:
        # VFA Tracking Case
        ax.plot(history['time'], history['S2'], 'b-', linewidth=2, label='VFA (S2)')
        ax.plot(history['time'], history['Setpoint'], 'k--', linewidth=2, label='Setpoint')
        ax.set_ylabel('Concentration [g/L]', fontsize=12, fontweight='bold')
        ax.set_title('VFA Tracking Performance', fontsize=14)
    elif s2_setpoint is not None:
         # VFA Tracking Case (implicit setpoint)
        ax.plot(history['time'], history['S2'], 'b-', linewidth=2, label='VFA (S2)')
        ax.axhline(y=s2_setpoint, color='k', linestyle='--', linewidth=2, label='Setpoint')
        ax.set_ylabel('Concentration [g/L]', fontsize=12, fontweight='bold')
        ax.set_title('VFA Tracking Performance', fontsize=14)
    else:
        # Standard Control Case (Show S1/VFA)
        ax.plot(history['time'], history['S1'], 'b-', linewidth=2, label='VFA (S1)')
        ax.set_ylabel('Concentration [g COD/L]', fontsize=12, fontweight='bold')
        ax.set_title('VFA Concentration', fontsize=14)
    
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Biogas
    ax = axes[1]
    ax.plot(history['time'], history['Q'], 'g-', linewidth=2, label='Biogas Production')
    ax.set_ylabel('Rate [L/d]', fontsize=12, fontweight='bold')
    ax.set_title('Biogas Production', fontsize=14)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Control Input
    ax = axes[2]
    ax.step(history['time'], history['D'], 'r-', where='post', label='Dilution Rate (D)')
    
    if d_max is not None:
        ax.axhline(y=d_max, color='r', linestyle=':', label='Max Constraint')
        ax.axhline(y=0, color='r', linestyle=':', label='Min Constraint')
        ax.set_ylim(-0.05, d_max + 0.1)
    else:
        ax.set_ylim(bottom=-0.05)
        
    ax.set_ylabel('Rate [1/d]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time [days]', fontsize=12, fontweight='bold')
    ax.set_title('Control Input & Constraints', fontsize=14)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    default_name = f"{title.lower().replace(' ', '_')}.png"
    final_path = resolve_save_path(save_path, save_plot, default_name)
    
    if final_path:
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {final_path}")
        
    if show:
        plt.show()
        
    return fig


__all__ = [
    'set_openad_style',
    'get_images_dir',
    'resolve_save_path',
    'plot_predictions',
    'plot_multi_output',
    'plot_training_curves',
    'plot_residuals',
    'plot_calibration_comparison',
    'quick_plot',
    'plot_mpc_results',
    'COLORS',
]
