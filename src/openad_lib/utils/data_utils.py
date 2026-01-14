"""
Data utilities for loading and preprocessing AD data.

Provides functions for loading various data formats and preprocessing
for use with OpenAD-lib models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
import os


def load_csv(
    path: str,
    time_column: Optional[str] = 'time',
    parse_dates: bool = False
) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        path: Path to CSV file
        time_column: Name of time column (None to skip parsing)
        parse_dates: Whether to parse date columns
    
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(path)
    
    if time_column and time_column in df.columns and parse_dates:
        df[time_column] = pd.to_datetime(df[time_column])
    
    return df


def load_excel(
    path: str,
    sheet_name: Optional[str] = None,
    time_column: Optional[str] = 'time'
) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Args:
        path: Path to Excel file
        sheet_name: Sheet name to load (None for first sheet)
        time_column: Name of time column
    
    Returns:
        DataFrame with loaded data
    """
    if sheet_name:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(path)
    
    return df


def preprocess_time_series(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    time_column: str = 'time',
    normalize: bool = True,
    fill_na: str = 'interpolate'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocess time series data for ML models.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Target column name
        time_column: Time column name
        normalize: Whether to standardize features
        fill_na: Method for handling NaN ('interpolate', 'drop', 'mean')
    
    Returns:
        Tuple of (X_array, y_array, preprocessing_info)
    """
    from sklearn.preprocessing import StandardScaler
    
    # Handle missing values
    if fill_na == 'interpolate':
        df = df.interpolate()
    elif fill_na == 'drop':
        df = df.dropna()
    elif fill_na == 'mean':
        df = df.fillna(df.mean())
    
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    info = {'feature_columns': feature_columns, 'target_column': target_column}
    
    if normalize:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        info['scaler_X'] = scaler_X
        info['scaler_y'] = scaler_y
    
    return X, y, info


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data preserving temporal order.
    
    Args:
        X: Feature array
        y: Target array  
        test_size: Fraction of data for testing
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def validate_adm1_input(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required ADM1 input columns.
    
    Args:
        df: Input DataFrame
        required_columns: List of required columns (uses default if None)
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    if required_columns is None:
        required_columns = [
            'S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
            'S_h2', 'S_ch4', 'S_IC', 'S_IN', 'S_I',
            'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa',
            'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
            'S_cation', 'S_anion'
        ]
    
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def get_sample_data_path(filename: str) -> str:
    """
    Get path to sample data file in package data directory.
    
    Args:
        filename: Name of data file
    
    Returns:
        Absolute path to data file
    """
    package_dir = Path(__file__).parent.parent
    data_dir = package_dir / 'data'
    return str(data_dir / filename)
