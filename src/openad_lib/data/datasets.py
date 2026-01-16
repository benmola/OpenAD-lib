"""
Dataset classes for openad_lib.

Provides PyTorch-compatible dataset wrappers for biogas and feedstock data.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple
from pathlib import Path


class BiogasDataset:
    """
    Dataset wrapper for biogas production data.
    
    Parameters
    ----------
    data : pd.DataFrame or str or Path
        DataFrame or path to CSV file containing biogas data
    target_column : str, default='Biogas'
        Name of column containing target values
    feature_columns : list of str, optional
        Feature column names. If None, uses all columns except target
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        target_column: str = 'Biogas',
        feature_columns: Optional[List[str]] = None
    ):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
        
        self.target_column = target_column
        
        # Determine feature columns
        if feature_columns is None:
            self.feature_columns = [c for c in self.data.columns if c != target_column]
        else:
            self.feature_columns = feature_columns
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        X = self.data[self.feature_columns].iloc[idx].values.astype(np.float32)
        y = self.data[self.target_column].iloc[idx]
        return X, y
    
    def get_features(self) -> np.ndarray:
        """Return all features as numpy array."""
        return self.data[self.feature_columns].values.astype(np.float32)
    
    def get_targets(self) -> np.ndarray:
        """Return all targets as numpy array."""
        return self.data[self.target_column].values.astype(np.float32)


class FeedstockDataset:
    """
    Dataset wrapper for feedstock composition data.
    
    Parameters
    ----------
    data : pd.DataFrame or str or Path
        DataFrame or path to CSV file containing feedstock data
    feedstock_columns : list of str, optional
        Feedstock column names. If None, auto-detects
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        feedstock_columns: Optional[List[str]] = None
    ):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
        
        # Auto-detect feedstock columns (exclude time, etc.)
        if feedstock_columns is None:
            exclude_cols = {'time', 'date', 'day', 'index'}
            self.feedstock_columns = [
                c for c in self.data.columns 
                if c.lower() not in exclude_cols
            ]
        else:
            self.feedstock_columns = feedstock_columns
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[self.feedstock_columns].iloc[idx].values.astype(np.float32)
    
    def get_ratios(self) -> np.ndarray:
        """Return feedstock ratios as numpy array."""
        return self.data[self.feedstock_columns].values.astype(np.float32)


class TimeSeriesDataset:
    """
    Dataset wrapper for time series data with lag features.
    
    Parameters
    ----------
    data : pd.DataFrame or str or Path
        DataFrame or path to CSV file
    features : list of str
        Feature column names
    target : str
        Target column name
    n_lags : int, default=1
        Number of lag timesteps to include
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        features: List[str],
        target: str,
        n_lags: int = 1
    ):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
        
        self.features = features
        self.target = target
        self.n_lags = n_lags
        
        # Create lag features
        self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences with lag features."""
        X_list, y_list = [], []
        
        data_values = self.data[self.features].values
        target_values = self.data[self.target].values
        
        for i in range(self.n_lags, len(self.data)):
            X_list.append(data_values[i-self.n_lags:i].flatten())
            y_list.append(target_values[i])
        
        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self.X[idx], self.y[idx]
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all X, y data."""
        return self.X, self.y


__all__ = [
    'BiogasDataset',
    'FeedstockDataset', 
    'TimeSeriesDataset'
]
