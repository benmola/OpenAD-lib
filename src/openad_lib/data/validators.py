"""
Data validation utilities for openad_lib.

This module provides functions for validating anaerobic digestion data
to ensure it meets requirements for different models and workflows.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings


def check_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True
) -> bool:
    """
    Check if DataFrame contains required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list of str
        List of required column names
    raise_error : bool, default=True
        Whether to raise ValueError if columns are missing
        
    Returns
    -------
    bool
        True if all required columns present
        
    Raises
    ------
    ValueError
        If required columns are missing and raise_error=True
        
    Examples
    --------
    >>> check_required_columns(df, ['time', 'Biogas'])
    True
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        msg = f"Missing required columns: {missing}"
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return False
    
    return True


def validate_influent_data(
    df: pd.DataFrame,
    model_type: str = 'adm1',
    strict: bool = False
) -> Dict[str, bool]:
    """
    Validate influent data for mechanistic models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Influent data to validate
    model_type : str, default='adm1'
        Model type: 'adm1' or 'am2'
    strict : bool, default=False
        If True, raise errors for any validation failures
        
    Returns
    -------
    dict
        Validation results with keys:
        - 'has_time': Time column present
        - 'has_required_columns': All required columns present
        - 'no_missing_values': No NaN values in critical columns
        - 'valid_ranges': All values in valid ranges
        
    Examples
    --------
    >>> results = validate_influent_data(influent_df, model_type='adm1')
    >>> if all(results.values()):
    ...     print("Data is valid!")
    """
    results = {
        'has_time': False,
        'has_required_columns': False,
        'no_missing_values': False,
        'valid_ranges': False
    }
    
    # Check for time column
    if 'time' in df.columns:
        results['has_time'] = True
    elif strict:
        raise ValueError("Time column required")
    
    # Define required columns by model type
    if model_type.lower() == 'adm1':
        required = ['S_su', 'S_aa', 'X_c', 'X_ch', 'X_pr', 'X_li']
    elif model_type.lower() == 'am2':
        required = ['S1in', 'D', 'pH']
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Check required columns
    try:
        check_required_columns(df, required, raise_error=strict)
        results['has_required_columns'] = True
    except ValueError:
        pass
    
    # Check for missing values
    if results['has_required_columns']:
        missing_count = df[required].isna().sum().sum()
        if missing_count == 0:
            results['no_missing_values'] = True
        elif strict:
            raise ValueError(f"Found {missing_count} missing values in required columns")
    
    # Check value ranges (all should be non-negative)
    if results['has_required_columns']:
        numeric_cols = df[required].select_dtypes(include=[np.number]).columns
        if (df[numeric_cols] >= 0).all().all():
            results['valid_ranges'] = True
        elif strict:
            raise ValueError("Negative values found in influent data")
    
    return results


def validate_measurement_data(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    strict: bool = False
) -> Dict[str, bool]:
    """
    Validate measurement/observation data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Measurement data to validate
    target_columns : list of str, optional
        Expected measurement columns (e.g., ['Biogas', 'VFA'])
    strict : bool, default=False
        If True, raise errors for validation failures
        
    Returns
    -------
    dict
        Validation results
        
    Examples
    --------
    >>> results = validate_measurement_data(
    ...     biogas_df,
    ...     target_columns=['Biogas', 'pH']
    ... )
    """
    results = {
        'has_time': False,
        'has_targets': False,
        'no_excessive_missing': False,
        'valid_ranges': False
    }
    
    # Check for time column
    if 'time' in df.columns:
        results['has_time'] = True
    elif strict:
        raise ValueError("Time column required in measurement data")
    
    # Check for target columns
    if target_columns:
        available = [col for col in target_columns if col in df.columns]
        if available:
            results['has_targets'] = True
        elif strict:
            raise ValueError(f"None of the target columns found: {target_columns}")
    else:
        results['has_targets'] = True  # No specific targets required
    
    # Check missing values (allow up to 10%)
    total_values = df.size
    missing_values = df.isna().sum().sum()
    missing_pct = (missing_values / total_values) * 100
    
    if missing_pct <= 10:
        results['no_excessive_missing'] = True
    elif strict:
        raise ValueError(f"Excessive missing values: {missing_pct:.1f}%")
    
    # Check numeric columns are non-negative
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        if (df[numeric_cols].dropna() >= 0).all().all():
            results['valid_ranges'] = True
        elif strict:
            warnings.warn("Negative values found in measurement data")
    
    return results


def validate_feedstock_data(
    df: pd.DataFrame,
    strict: bool = False
) -> Dict[str, bool]:
    """
    Validate feedstock composition data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feedstock data to validate
    strict : bool, default=False
        If True, raise errors for validation failures
        
    Returns
    -------
    dict
        Validation results
        
    Examples
    --------
    >>> results = validate_feedstock_data(feedstock_df)
    """
    results = {
        'has_time': False,
        'has_feedstocks': False,
        'ratios_sum_to_100': False,
        'no_negative_values': False
    }
    
    # Check for time column
    if 'time' in df.columns:
        results['has_time'] = True
    
    # Check for feedstock columns (exclude 'time')
    feedstock_cols = [col for col in df.columns if col != 'time']
    if feedstock_cols:
        results['has_feedstocks'] = True
    elif strict:
        raise ValueError("No feedstock columns found")
    
    # Check if ratios sum to ~100 (allowing small tolerance)
    if results['has_feedstocks']:
        row_sums = df[feedstock_cols].sum(axis=1)
        if ((row_sums >= 99) & (row_sums <= 101)).all():
            results['ratios_sum_to_100'] = True
        elif strict:
            raise ValueError("Feedstock ratios should sum to 100%")
    
    # Check for negative values
    if results['has_feedstocks']:
        if (df[feedstock_cols] >= 0).all().all():
            results['no_negative_values'] = True
        elif strict:
            raise ValueError("Negative values found in feedstock data")
    
    return results


__all__ = [
    'check_required_columns',
    'validate_influent_data',
    'validate_measurement_data',
    'validate_feedstock_data'
]
