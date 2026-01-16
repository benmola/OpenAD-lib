"""
Input validation utilities for openad_lib.

Provides validation functions to check data quality, parameter bounds,
and physical constraints before model execution.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union


def validate_influent_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    allow_nan: bool = False
) -> None:
    """
    Validate influent data for AD models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Influent dataframe to validate
    required_columns : list, optional
        Required column names. If None, uses default minimal set.
    allow_nan : bool, default=False
        Whether to allow NaN values
        
    Raises
    ------
    ValueError
        If validation fails
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'time': [0, 1, 2], 'S1_in': [10, 12, 11]})
    >>> validate_influent_data(df, required_columns=['time', 'S1_in'])
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Default required columns if not specified
    if required_columns is None:
        required_columns = ['time']
    
    # Check for missing columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    if not allow_nan:
        nan_cols = df[required_columns].columns[df[required_columns].isnull().any()].tolist()
        if nan_cols:
            raise ValueError(f"NaN values found in columns: {nan_cols}")
    
    # Check time column
    if 'time' in df.columns:
        if not np.all(np.diff(df['time']) >= 0):
            raise ValueError("Time column must be monotonically increasing")


def validate_state_bounds(
    states: Union[np.ndarray, Dict[str, np.ndarray]],
    model: str = 'AM2',
    strict: bool = True
) -> None:
    """
    Check if model states satisfy physical constraints.
    
    Parameters
    ----------
    states : np.ndarray or dict
        Model states (concentrations, biomass, etc.)
        Can be array or dict of arrays
    model : str, default='AM2'
        Model type: 'AM2', 'ADM1'
    strict : bool, default=True
        If True, raises error. If False, issues warning.
        
    Raises
    ------
    ValueError
        If states violate physical constraints (strict=True)
        
    Examples
    --------
    >>> states = np.array([5.0, 0.5, 2.0, 0.3])  # S1, X1, S2, X2
    >>> validate_state_bounds(states, model='AM2')
    """
    if isinstance(states, dict):
        # Check each state variable
        for name, values in states.items():
            _check_non_negative(values, name, strict)
    else:
        # Array input
        states = np.asarray(states)
        _check_non_negative(states, 'states', strict)
    
    # Model-specific checks
    if model.upper() == 'AM2':
        _validate_am2_states(states, strict)
    elif model.upper() == 'ADM1':
        _validate_adm1_states(states, strict)


def _check_non_negative(
    values: np.ndarray,
    name: str,
    strict: bool
) -> None:
    """Check for negative values (concentrations must be >= 0)."""
    values = np.asarray(values)
    
    if np.any(values < 0):
        msg = f"Negative values detected in {name}: min={np.min(values):.4f}"
        if strict:
            raise ValueError(msg)
        else:
            import warnings
            warnings.warn(msg)


def _validate_am2_states(states, strict: bool) -> None:
    """AM2-specific validation (4 states: S1, X1, S2, X2)."""
    if isinstance(states, np.ndarray):
        if states.ndim == 1:
            if len(states) != 4:
                raise ValueError(f"AM2 requires 4 states, got {len(states)}")
        elif states.ndim == 2:
            if states.shape[1] != 4:
                raise ValueError(f"AM2 requires 4 states, got {states.shape[1]}")


def _validate_adm1_states(states, strict: bool) -> None:
    """ADM1-specific validation (35-38 states depending on implementation)."""
    if isinstance(states, np.ndarray):
        if states.ndim == 1:
            if len(states) < 35:
                raise ValueError(f"ADM1 requires at least 35 states, got {len(states)}")


def validate_params(
    params: Dict[str, float],
    bounds: Dict[str, tuple],
    strict: bool = True
) -> None:
    """
    Validate model parameters against physical bounds.
    
    Parameters
    ----------
    params : dict
        Parameter values to validate
    bounds : dict
        Dictionary of (min, max) tuples for each parameter
    strict : bool, default=True
        If True, raises error. If False, issues warning.
        
    Raises
    ------
    ValueError
        If parameters are out of bounds (strict=True)
        
    Examples
    --------
    >>> params = {'m1': 0.2, 'K1': 15.0}
    >>> bounds = {'m1': (0.05, 0.5), 'K1': (5.0, 30.0)}
    >>> validate_params(params, bounds)
    """
    violations = []
    
    for param_name, value in params.items():
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            if not (min_val <= value <= max_val):
                violations.append(
                    f"{param_name}={value:.4f} outside [{min_val}, {max_val}]"
                )
    
    if violations:
        msg = "Parameter validation failed:\n  " + "\n  ".join(violations)
        if strict:
            raise ValueError(msg)
        else:
            import warnings
            warnings.warn(msg)


def validate_control_input(
    u: Union[float, np.ndarray],
    u_min: float = 0.0,
    u_max: float = 1.0,
    name: str = 'control input'
) -> None:
    """
    Validate control input is within actuator limits.
    
    Parameters
    ----------
    u : float or np.ndarray
        Control input(s)
    u_min : float, default=0.0
        Minimum allowed value
    u_max : float, default=1.0
        Maximum allowed value
    name : str, default='control input'
        Variable name for error messages
        
    Raises
    ------
    ValueError
        If control input is out of bounds
        
    Examples
    --------
    >>> validate_control_input(0.3, u_min=0.0, u_max=0.5)  # OK
    >>> validate_control_input(0.8, u_min=0.0, u_max=0.5)  # Raises error
    """
    u = np.asarray(u)
    
    if np.any(u < u_min):
        raise ValueError(f"{name} below minimum: min={np.min(u):.4f} < {u_min}")
    
    if np.any(u > u_max):
        raise ValueError(f"{name} above maximum: max={np.max(u):.4f} > {u_max}")


def validate_time_series(
    data: Union[pd.DataFrame, np.ndarray],
    min_length: int = 10,
    check_monotonic: bool = True
) -> None:
    """
    Validate time series data for ML models.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Time series data
    min_length : int, default=10
        Minimum required sequence length
    check_monotonic : bool, default=True
        Whether to verify time is monotonically increasing
        
    Raises
    ------
    ValueError
        If validation fails
        
    Examples
    --------
    >>> df = pd.DataFrame({'time': [0, 1, 2], 'value': [1, 2, 3]})
    >>> validate_time_series(df, min_length=3)
    """
    if isinstance(data, pd.DataFrame):
        n_samples = len(data)
        if 'time' in data.columns and check_monotonic:
            if not np.all(np.diff(data['time']) >= 0):
                raise ValueError("Time must be monotonically increasing")
    else:
        data = np.asarray(data)
        n_samples = data.shape[0]
    
    if n_samples < min_length:
        raise ValueError(f"Insufficient data: {n_samples} < {min_length} samples")


def validate_train_test_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Validate train/test split dimensions match.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices
    y_train, y_test : np.ndarray
        Target arrays
        
    Raises
    ------
    ValueError
        If dimensions don't match
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) "
                        f"have different number of samples")
    
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]}) "
                        f"have different number of samples")
    
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"X_train ({X_train.shape[1]} features) and X_test "
                        f"({X_test.shape[1]} features) have different dimensions")


__all__ = [
    'validate_influent_data',
    'validate_state_bounds',
    'validate_params',
    'validate_control_input',
    'validate_time_series',
    'validate_train_test_split',
]
