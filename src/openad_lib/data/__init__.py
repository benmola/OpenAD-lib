"""
Data loading and management utilities for openad_lib.

This module provides standardized data loading, validation, and dataset classes
for working with anaerobic digestion data.

Available functions:
    - load_feedstock_data: Load feedstock composition data
    - load_biogas_data: Load biogas production measurements
    - load_influent_data: Load influent characterization data

Available classes:
    - BiogasDataset: Dataset wrapper for biogas production data
    - FeedstockDataset: Dataset wrapper for feedstock data
"""

from .loaders import (
    load_feedstock_data,
    load_biogas_data,
    load_influent_data,
    load_csv_data
)

from .validators import (
    validate_influent_data,
    validate_measurement_data,
    validate_feedstock_data,
    check_required_columns
)

from .datasets import (
    BiogasDataset,
    FeedstockDataset,
    TimeSeriesDataset
)

__all__ = [
    # Loaders
    'load_feedstock_data',
    'load_biogas_data',
    'load_influent_data',
    'load_csv_data',
    
    # Validators
    'validate_influent_data',
    'validate_measurement_data',
    'validate_feedstock_data',
    'check_required_columns',
    
    # Datasets
    'BiogasDataset',
    'FeedstockDataset',
    'TimeSeriesDataset'
]
