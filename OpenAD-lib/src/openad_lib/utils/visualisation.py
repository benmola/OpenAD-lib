"""
Visualization utilities for AD process data.

Provides plotting functions for simulation results, model comparisons,
and uncertainty visualization.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_simulation_results(
    results: pd.DataFrame,
    variables: Optional[List[str]] = None,
    time_column: str = 'time',
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> None:
    """
    Plot ADM1 simulation results.
    
    Args:
        results: DataFrame with simulation results
        variables: List of variables to plot (None for defaults)
        time_column: Name of time column
        figsize: Figure size
        title: Optional plot title
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed. Cannot plot.")
        return
    
    if variables is None:
        variables = ['S_ac', 'S_pro', 'pH']
    
    n_vars = len(variables)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i, var in enumerate(variables):
        row, col = i // n_cols, i % n_cols
        if var in results.columns:
            axes[row, col].plot(results[time_column], results[var])
            axes[row, col].set_xlabel('Time (days)')
            axes[row, col].set_ylabel(var)
            axes[row, col].set_title(var)
            axes[row, col].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()


def plot_biogas_production(
    gas_flow: pd.DataFrame,
    actual_data: Optional[pd.DataFrame] = None,
    actual_time_col: str = 'time',
    actual_biogas_col: str = 'Biogas (m3/day)',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot biogas production comparison.
    
    Args:
        gas_flow: DataFrame with model gas flow results
        actual_data: Optional DataFrame with actual measurements
        actual_time_col: Time column in actual data
        actual_biogas_col: Biogas column in actual data
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(gas_flow['time'], gas_flow['q_gas'], 'b-', label='Model', linewidth=2)
    
    if actual_data is not None and actual_biogas_col in actual_data.columns:
        ax.plot(
            actual_data[actual_time_col], 
            actual_data[actual_biogas_col],
            'ro--', label='Measured', markersize=4, alpha=0.7
        )
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Biogas Flow (m³/day)', fontsize=12)
    ax.set_title('Biogas Production', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_uncertainty(
    time: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: Optional[np.ndarray] = None,
    label: str = 'Prediction',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot predictions with uncertainty bands.
    
    Args:
        time: Time array
        mean: Mean predictions
        lower: Lower confidence bound
        upper: Upper confidence bound
        actual: Optional actual values
        label: Label for predictions
        ylabel: Y-axis label
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time, mean, 'b-', label=label, linewidth=2)
    ax.fill_between(time, lower, upper, alpha=0.3, color='blue', label='95% CI')
    
    if actual is not None:
        ax.plot(time, actual, 'ro', label='Actual', markersize=4)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    models_results: Dict[str, pd.DataFrame],
    variable: str,
    actual_data: Optional[pd.DataFrame] = None,
    time_column: str = 'time',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Compare multiple model predictions.
    
    Args:
        models_results: Dict of {model_name: results_df}
        variable: Variable to compare
        actual_data: Optional actual data
        time_column: Time column name
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_results)))
    
    for (name, results), color in zip(models_results.items(), colors):
        if variable in results.columns:
            ax.plot(results[time_column], results[variable], 
                   label=name, color=color, linewidth=2)
    
    if actual_data is not None and variable in actual_data.columns:
        ax.plot(actual_data[time_column], actual_data[variable],
               'ko', label='Actual', markersize=4)
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel(variable, fontsize=12)
    ax.set_title(f'Model Comparison: {variable}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
