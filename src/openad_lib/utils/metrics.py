"""
Unified metrics module for model evaluation.

Provides consistent metric calculations across all models in openad_lib.
"""

import numpy as np
from typing import Dict, Optional, Union


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    metrics: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metrics : list, optional
        List of metrics to compute. If None, computes all.
        Options: ['rmse', 'mae', 'r2', 'mape', 'nrmse']
        
    Returns
    -------
    results : dict
        Dictionary with computed metrics
        
    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    >>> metrics = compute_metrics(y_true, y_pred)
    >>> print(f"RMSE: {metrics['RMSE']:.3f}")
    """
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Compute all metrics
    residuals = y_true - y_pred
    
    results = {}
    
    # RMSE (Root Mean Square Error)
    if metrics is None or 'rmse' in [m.lower() for m in metrics]:
        results['RMSE'] = float(np.sqrt(np.mean(residuals**2)))
    
    # MAE (Mean Absolute Error)
    if metrics is None or 'mae' in [m.lower() for m in metrics]:
        results['MAE'] = float(np.mean(np.abs(residuals)))
    
    # R² (Coefficient of Determination)
    if metrics is None or 'r2' in [m.lower() for m in metrics]:
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        results['R2'] = float(1 - (ss_res / (ss_tot + 1e-10)))
    
    # MAPE (Mean Absolute Percentage Error)
    if metrics is None or 'mape' in [m.lower() for m in metrics]:
        # Avoid division by zero
        mask = np.abs(y_true) > 1e-10
        if np.any(mask):
            mape = np.mean(np.abs(residuals[mask] / y_true[mask])) * 100
            results['MAPE'] = float(mape)
        else:
            results['MAPE'] = np.nan
    
    # NRMSE (Normalized RMSE)
    if metrics is None or 'nrmse' in [m.lower() for m in metrics]:
        mean_true = np.mean(y_true)
        if mean_true > 1e-10:
            results['NRMSE'] = float(results.get('RMSE', np.sqrt(np.mean(residuals**2))) / mean_true)
        else:
            results['NRMSE'] = np.nan
    
    return results


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Square Error.
    
    NRMSE = RMSE / mean(y_true)
    
    Useful for comparing errors across different scales.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    nrmse : float
        Normalized RMSE value
        
    Examples
    --------
    >>> y_true = np.array([10, 20, 30, 40])
    >>> y_pred = np.array([12, 19, 31, 38])
    >>> nrmse = normalized_rmse(y_true, y_pred)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_true = np.mean(y_true)
    
    if mean_true < 1e-10:
        raise ValueError("Cannot compute NRMSE: mean of y_true is too close to zero")
    
    return float(rmse / mean_true)


def print_metrics(
    metrics: Dict[str, float],
    title: Optional[str] = None,
    precision: int = 4
) -> None:
    """
    Print metrics in a formatted table.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values
    title : str, optional
        Title to display above metrics
    precision : int, default=4
        Number of decimal places
        
    Examples
    --------
    >>> metrics = {'RMSE': 0.123, 'MAE': 0.098, 'R2': 0.95}
    >>> print_metrics(metrics, title="Model Performance")
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    else:
        print("\nMetrics:")
        print("=" * 40)
    
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            print(f"  {name:10s}: {value:.{precision}f}")
        else:
            print(f"  {name:10s}: {value}")
    print()


def compute_multi_output_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for multi-output predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_outputs)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_outputs)
    output_names : list, optional
        Names for each output dimension
        
    Returns
    -------
    results : dict
        Nested dictionary: {output_name: {metric: value}}
        
    Examples
    --------
    >>> y_true = np.random.rand(100, 3)
    >>> y_pred = y_true + np.random.randn(100, 3) * 0.1
    >>> metrics = compute_multi_output_metrics(
    ...     y_true, y_pred, 
    ...     output_names=['SCOD', 'VFA', 'Biogas']
    ... )
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_outputs = y_true.shape[1]
    
    if output_names is None:
        output_names = [f"Output_{i}" for i in range(n_outputs)]
    
    if len(output_names) != n_outputs:
        raise ValueError(f"Number of output_names ({len(output_names)}) "
                        f"doesn't match n_outputs ({n_outputs})")
    
    results = {}
    for i, name in enumerate(output_names):
        results[name] = compute_metrics(y_true[:, i], y_pred[:, i])
    
    return results


def aggregate_metrics(
    multi_output_metrics: Dict[str, Dict[str, float]],
    method: str = 'mean'
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple outputs.
    
    Parameters
    ----------
    multi_output_metrics : dict
        Output from compute_multi_output_metrics()
    method : str, default='mean'
        Aggregation method: 'mean', 'median', 'min', 'max'
        
    Returns
    -------
    aggregated : dict
        Single set of aggregated metrics
        
    Examples
    --------
    >>> multi_metrics = {
    ...     'SCOD': {'RMSE': 0.5, 'R2': 0.9},
    ...     'VFA': {'RMSE': 0.3, 'R2': 0.95}
    ... }
    >>> avg_metrics = aggregate_metrics(multi_metrics)
    """
    # Collect all metric names
    all_metric_names = set()
    for output_metrics in multi_output_metrics.values():
        all_metric_names.update(output_metrics.keys())
    
    aggregated = {}
    
    for metric_name in all_metric_names:
        values = []
        for output_metrics in multi_output_metrics.values():
            if metric_name in output_metrics:
                val = output_metrics[metric_name]
                if not np.isnan(val):
                    values.append(val)
        
        if values:
            if method == 'mean':
                aggregated[metric_name] = np.mean(values)
            elif method == 'median':
                aggregated[metric_name] = np.median(values)
            elif method == 'min':
                aggregated[metric_name] = np.min(values)
            elif method == 'max':
                aggregated[metric_name] = np.max(values)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        else:
            aggregated[metric_name] = np.nan
    
    return aggregated


__all__ = [
    'compute_metrics',
    'normalized_rmse',
    'print_metrics',
    'compute_multi_output_metrics',
    'aggregate_metrics',
]
