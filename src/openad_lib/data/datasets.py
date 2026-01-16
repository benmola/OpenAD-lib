"""
Dataset classes for openad_lib.

This module provides dataset wrapper classes for working with anaerobic digestion data
in a standardized way.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass

from .loaders import load_csv_data
from .validators import validate_measurement_data, validate_feedstock_data


@dataclass
class BiogasDataset:
    """
    Dataset wrapper for biogas production measurements.
    
    Attributes
    ----------
    data : pd.DataFrame
        Raw measurement data
    time : np.ndarray
        Time points
    biogas : np.ndarray
        Biogas production values
    metadata : dict
        Additional metadata about the dataset
        
    Examples
    --------
    >>> dataset = BiogasDataset.from_csv('biogas_data.csv')
    >>> print(f"Dataset has {len(dataset)} samples")
    >>> time, biogas = dataset.get_arrays()
    """
    
    data: pd.DataFrame
    time_column: str = 'time'
    biogas_column: str = 'Biogas'
    metadata: dict = None
    
    def __post_init__(self):
        """Validate and extract arrays."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validate data
        self.time = self.data[self.time_column].values
        
        # Find biogas column (flexible matching)
        biogas_cols = [col for col in self.data.columns 
                      if self.biogas_column.lower() in col.lower()]
        if biogas_cols:
            self.biogas = self.data[biogas_cols[0]].values
            self.biogas_column = biogas_cols[0]
        elif self.biogas_column in self.data.columns:
            self.biogas = self.data[self.biogas_column].values
        else:
            raise ValueError(f"Biogas column '{self.biogas_column}' not found")
    
    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        time_column: str = 'time',
        biogas_column: str = 'Biogas',
        **kwargs
    ):
        """
        Load dataset from CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
        time_column : str
            Name of time column
        biogas_column : str
            Name of biogas column
        **kwargs
            Additional arguments for pd.read_csv
            
        Returns
        -------
        BiogasDataset
            Loaded dataset
        """
        data = load_csv_data(filepath, **kwargs)
        return cls(data, time_column, biogas_column)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time and biogas arrays.
        
        Returns
        -------
        time : np.ndarray
            Time points
        biogas : np.ndarray
            Biogas values
        """
        return self.time, self.biogas
    
    def split(
        self,
        train_fraction: float = 0.8
    ) -> Tuple['BiogasDataset', 'BiogasDataset']:
        """
        Split dataset into train and test sets.
        
        Parameters
        ----------
        train_fraction : float
            Fraction of data for training (0-1)
            
        Returns
        -------
        train_dataset : BiogasDataset
            Training dataset
        test_dataset : BiogasDataset
            Testing dataset
        """
        split_idx = int(len(self.data) * train_fraction)
        
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()
        
        train_dataset = BiogasDataset(
            train_data, self.time_column, self.biogas_column,
            metadata={'split': 'train', **self.metadata}
        )
        test_dataset = BiogasDataset(
            test_data, self.time_column, self.biogas_column,
            metadata={'split': 'test', **self.metadata}
        )
        
        return train_dataset, test_dataset
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BiogasDataset(n_samples={len(self)}, time_range=[{self.time.min():.1f}, {self.time.max():.1f}])"


@dataclass
class FeedstockDataset:
    """
    Dataset wrapper for feedstock composition data.
    
    Attributes
    ----------
    data : pd.DataFrame
        Raw feedstock data
    time : np.ndarray
        Time points
    feedstocks : List[str]
        List of feedstock names
    ratios : np.ndarray
        Feedstock ratio matrix (n_samples x n_feedstocks)
        
    Examples
    --------
    >>> dataset = FeedstockDataset.from_csv('Feed_Data.csv')
    >>> print(f"Feedstocks: {dataset.feedstocks}")
    >>> ratios = dataset.get_ratios()
    """
    
    data: pd.DataFrame
    time_column: str = 'time'
    metadata: dict = None
    
    def __post_init__(self):
        """Extract feedstock information."""
        if self.metadata is None:
            self.metadata = {}
        
        # Get time
        if self.time_column in self.data.columns:
            self.time = self.data[self.time_column].values
        else:
            self.time = np.arange(len(self.data))
        
        # Get feedstock columns (all except time)
        self.feedstocks = [col for col in self.data.columns if col != self.time_column]
        self.ratios = self.data[self.feedstocks].values
    
    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        time_column: str = 'time',
        **kwargs
    ):
        """Load feedstock dataset from CSV file."""
        data = load_csv_data(filepath, **kwargs)
        return cls(data, time_column)
    
    def get_ratios(self, feedstock_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get feedstock ratios.
        
        Parameters
        ----------
        feedstock_names : list of str, optional
            Specific feedstocks to extract. If None, returns all.
            
        Returns
        -------
        np.ndarray
            Feedstock ratio matrix
        """
        if feedstock_names is None:
            return self.ratios
        
        indices = [self.feedstocks.index(name) for name in feedstock_names]
        return self.ratios[:, indices]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FeedstockDataset(n_samples={len(self)}, n_feedstocks={len(self.feedstocks)})"


@dataclass
class TimeSeriesDataset:
    """
    Generic time series dataset for ML models.
    
    Attributes
    ----------
    data : pd.DataFrame
        Raw time series data
    features : List[str]
        Feature column names
    targets : List[str]
        Target column names
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target matrix
        
    Examples
    --------
    >>> dataset = TimeSeriesDataset.from_csv(
    ...     'timeseries.csv',
    ...     features=['Maize', 'Manure'],
    ...     targets=['Biogas']
    ... )
    >>> X_train, y_train = dataset.get_train_data()
    """
    
    data: pd.DataFrame
    features: List[str]
    targets: List[str]
    metadata: dict = None
    
    def __post_init__(self):
        """Extract feature and target arrays."""
        if self.metadata is None:
            self.metadata = {}
        
        self.X = self.data[self.features].values
        self.y = self.data[self.targets].values
    
    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        features: List[str],
        targets: List[str],
        **kwargs
    ):
        """Load time series dataset from CSV file."""
        data = load_csv_data(filepath, **kwargs)
        return cls(data, features, targets)
    
    def split(
        self,
        train_fraction: float = 0.8
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split into train and test sets.
        
        Returns
        -------
        train_data : tuple
            (X_train, y_train)
        test_data : tuple
            (X_test, y_test)
        """
        split_idx = int(len(self.data) * train_fraction)
        
        X_train = self.X[:split_idx]
        y_train = self.y[:split_idx]
        X_test = self.X[split_idx:]
        y_test = self.y[split_idx:]
        
        return (X_train, y_train), (X_test, y_test)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TimeSeriesDataset(n_samples={len(self)}, n_features={len(self.features)}, n_targets={len(self.targets)})"


__all__ = [
    'BiogasDataset',
    'FeedstockDataset',
    'TimeSeriesDataset'
]
