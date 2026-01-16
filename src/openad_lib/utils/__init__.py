"""
Utilities module for openad_lib.

Provides data loading, metrics computation, validation, and plotting utilities.
"""

# Metrics module (new)
from openad_lib.utils.metrics import (
    compute_metrics,
    normalized_rmse,
    print_metrics,
    compute_multi_output_metrics,
    aggregate_metrics,
)

# Validation module (new)
from openad_lib.utils.validation import (
    validate_influent_data,
    validate_state_bounds,
    validate_params,
    validate_control_input,
    validate_time_series,
    validate_train_test_split,
)

# Data utilities (existing - backward compatibility)
from openad_lib.utils.data_utils import (
    load_csv,
    load_excel,
    preprocess_time_series,
    train_test_split_temporal,
    validate_adm1_input,
    get_sample_data_path,
)

# Plot utilities (existing)
try:
    from openad_lib.utils.plot_utils import (
        plot_time_series,
        plot_comparison,
    )
except ImportError:
    # Plotting utilities may not be available yet
    pass


__all__ = [
    # Metrics
    'compute_metrics',
    'normalized_rmse',
    'print_metrics',
    'compute_multi_output_metrics',
    'aggregate_metrics',
    # Validation
    'validate_influent_data',
    'validate_state_bounds',
    'validate_params',
    'validate_control_input',
    'validate_time_series',
    'validate_train_test_split',
    # Data utilities
    'load_csv',
    'load_excel',
    'preprocess_time_series',
    'train_test_split_temporal',
    'validate_adm1_input',
    'get_sample_data_path',
    # Plotting (if available)
    'plot_time_series',
    'plot_comparison',
]
