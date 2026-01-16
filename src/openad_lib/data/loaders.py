"""
Data loading utilities for openad_lib.

This module provides functions for loading various types of anaerobic digestion data
from CSV files and other formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
import warnings

from openad_lib.config import config


def load_csv_data(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Load data from CSV file with standard error handling.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    **kwargs
        Additional arguments passed to pd.read_csv()
        
    Returns
    -------
    pd.DataFrame
        Loaded data
        
    Examples
    --------
    >>> data = load_csv_data('biogas_data.csv')
    >>> data = load_csv_data('data.csv', sep=';', decimal=',')
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, **kwargs)
        if config.verbose:
            print(f"Loaded {len(df)} rows from {filepath.name}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file {filepath}: {e}")


def load_feedstock_data(
    filepath: Union[str, Path],
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load feedstock composition data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to feedstock data CSV file
    required_columns : list of str, optional
        Required column names to validate
        
    Returns
    -------
    pd.DataFrame
        Feedstock data with validated columns
        
    Examples
    --------
    >>> feedstock = load_feedstock_data('Feed_Data.csv')
    >>> feedstock = load_feedstock_data(
    ...     'feed.csv',
    ...     required_columns=['time', 'Maize', 'Manure']
    ... )
    """
    df = load_csv_data(filepath)
    
    # Default required columns for feedstock data
    if required_columns is None:
        required_columns = ['time']
    
    # Check for required columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure time column is numeric
    if 'time' in df.columns:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
    
    return df


def load_biogas_data(
    filepath: Union[str, Path],
    biogas_column: str = 'Biogas',
    time_column: str = 'time'
) -> pd.DataFrame:
    """
    Load biogas production measurement data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to biogas data CSV file
    biogas_column : str, default='Biogas'
        Name of column containing biogas production values
    time_column : str, default='time'
        Name of column containing time values
        
    Returns
    -------
    pd.DataFrame
        Biogas measurement data
        
    Examples
    --------
    >>> biogas = load_biogas_data('Biogas_Plant_Outputs.csv')
    >>> biogas = load_biogas_data(
    ...     'measurements.csv',
    ...     biogas_column='q_gas',
    ...     time_column='days'
    ... )
    """
    df = load_csv_data(filepath)
    
    # Check for time column
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in data")
    
    # Check for biogas column (flexible matching)
    biogas_cols = [col for col in df.columns if biogas_column.lower() in col.lower()]
    if not biogas_cols and biogas_column not in df.columns:
        warnings.warn(
            f"Biogas column '{biogas_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Ensure numeric types
    df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
    for col in biogas_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_influent_data(
    filepath: Union[str, Path],
    validate: bool = True
) -> pd.DataFrame:
    """
    Load influent characterization data (e.g., from ACoD preprocessing).
    
    Parameters
    ----------
    filepath : str or Path
        Path to influent data CSV file
    validate : bool, default=True
        Whether to validate required ADM1 influent columns
        
    Returns
    -------
    pd.DataFrame
        Influent characterization data
        
    Examples
    --------
    >>> influent = load_influent_data('influent_data.csv')
    """
    df = load_csv_data(filepath)
    
    if validate:
        # Check for typical ADM1 influent columns
        expected_cols = ['time', 'S_su', 'S_aa', 'X_c', 'X_ch', 'X_pr', 'X_li']
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            warnings.warn(
                f"Some expected influent columns missing: {missing}. "
                f"This may be intentional if using a different model."
            )
    
    return df


def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """
    Load sample datasets included with openad_lib.
    
    Parameters
    ----------
    dataset_name : str
        Name of sample dataset:
        - 'feedstock': Sample feedstock composition
        - 'biogas': Sample biogas measurements
        - 'am2_lab': Sample AM2 laboratory data
        - 'lstm_timeseries': Sample LSTM training data
        
    Returns
    -------
    pd.DataFrame
        Sample dataset
        
    Examples
    --------
    >>> feedstock = load_sample_data('feedstock')
    >>> biogas = load_sample_data('biogas')
    """
    data_dir = config.data_dir
    
    dataset_files = {
        'feedstock': 'feedstock/Feed_Data.csv',
        'biogas': 'Biogas_Plant_Outputs.csv',
        'am2_lab': 'sample_AM2_Lab_data.csv',
        'lstm_timeseries': 'sample_LSTM_timeseries.csv',
        'mtgp': 'sample_ad_process_data.csv'
    }
    
    if dataset_name not in dataset_files:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(dataset_files.keys())}"
        )
    
    filepath = data_dir / dataset_files[dataset_name]
    return load_csv_data(filepath)


__all__ = [
    'load_csv_data',
    'load_feedstock_data',
    'load_biogas_data',
    'load_influent_data',
    'load_sample_data'
]
