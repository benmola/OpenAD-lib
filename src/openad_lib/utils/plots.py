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
    show: bool = True
) -> plt.Figure:
    """
    Plot predictions vs actual values with optional confidence intervals.
    
    Automatically handles train/test split visualization and uncertainty.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    x : array-like, optional
        X-axis values (default: indices)
    y_lower : array-like, optional
        Lower confidence bound
    y_upper : array-like, optional
        Upper confidence bound
    train_indices : array-like, optional
        Indices for training data
    test_indices : array-like, optional
        Indices for test data
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    save_path : str or Path, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
        
    Examples
    --------
    >>> import openad_lib as openad
    >>> openad.plots.plot_predictions(y_true, y_pred, title="LSTM Results")
    >>> 
    >>> # With confidence intervals
    >>> openad.plots.plot_predictions(
    ...     y_true, y_pred, y_lower=lower, y_upper=upper,
    ...     save_path="results.png"
    ... )
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
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
    xlabel: str = "Time",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot multiple outputs in subplots (like the example figure).
    
    Perfect for multi-task models (MTGP, multi-output LSTM, etc.)
    
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_outputs)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_outputs)
    x : np.ndarray, optional
        X-axis values
    y_lower : np.ndarray, optional
        Lower confidence bounds, shape (n_samples, n_outputs)
    y_upper : np.ndarray, optional
        Upper confidence bounds, shape (n_samples, n_outputs)
    train_indices : np.ndarray, optional
        Indices for training data points
    test_indices : np.ndarray, optional
        Indices for test data points
    output_names : list of str, optional
        Names for each output
    xlabel : str
        X-axis label (only shown on bottom subplot)
    save_path : str or Path, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
        
    Examples
    --------
    >>> import openad_lib as openad
    >>> # Multi-task GP results
    >>> openad.plots.plot_multi_output(
    ...     Y_test, predictions, 
    ...     output_names=['SCODout', 'VFAout', 'Biogas'],
    ...     y_lower=lower, y_upper=upper
    ... )
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
    
    # **FIX: Sort all data by x-axis for smooth lines**
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
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training History",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Parameters
    ----------
    train_losses : list of float
        Training losses per epoch
    val_losses : list of float, optional
        Validation losses per epoch
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot residuals (prediction errors) analysis.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
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
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
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


__all__ = [
    'set_openad_style',
    'plot_predictions',
    'plot_multi_output',
    'plot_training_curves',
    'plot_residuals',
    'quick_plot',
    'COLORS',
]
