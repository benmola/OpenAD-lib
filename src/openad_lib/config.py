"""
Configuration management for openad_lib.

This module provides centralized configuration for paths, model defaults,
numerical settings, and logging options.

Example:
    >>> from openad_lib.config import config
    >>> config.default_device = 'cuda'
    >>> config.verbose = False
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    """
    Global configuration for openad_lib.
    
    Attributes
    ----------
    data_dir : Path
        Directory containing package data files
    cache_dir : Path, optional
        Directory for caching intermediate results
    default_device : str
        Default device for ML models ('cpu' or 'cuda')
    random_seed : int
        Random seed for reproducibility
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    verbose : bool
        Whether to print progress information
    ode_solver : str
        Default ODE solver for mechanistic models
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver
    max_workers : int
        Maximum number of parallel workers for optimization
    
    Examples
    --------
    >>> from openad_lib.config import config
    >>> config.verbose = False
    >>> config.default_device = 'cuda'
    >>> config.ode_solver = 'RK45'
    """
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / 'data')
    cache_dir: Optional[Path] = None
    
    # Model defaults
    default_device: str = 'cpu'  # 'cpu' or 'cuda'
    random_seed: int = 42
    
    # Logging
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    verbose: bool = True
    
    # Numerical settings for ODE solvers
    ode_solver: str = 'LSODA'  # 'LSODA', 'RK45', 'BDF', etc.
    rtol: float = 1e-6  # Relative tolerance
    atol: float = 1e-8  # Absolute tolerance
    
    # Optimization settings
    max_workers: int = 1  # Number of parallel workers
    
    def __post_init__(self):
        """Initialize paths and validate settings."""
        # Ensure data_dir exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Create cache_dir if specified
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate device
        if self.default_device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {self.default_device}. Must be 'cpu' or 'cuda'")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}. Must be one of {valid_levels}")
    
    def set_cache_dir(self, path: str):
        """
        Set cache directory for storing intermediate results.
        
        Parameters
        ----------
        path : str
            Path to cache directory
        """
        self.cache_dir = Path(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def reset_to_defaults(self):
        """Reset all configuration to default values."""
        self.__init__()
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            'data_dir': str(self.data_dir),
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
            'default_device': self.default_device,
            'random_seed': self.random_seed,
            'log_level': self.log_level,
            'verbose': self.verbose,
            'ode_solver': self.ode_solver,
            'rtol': self.rtol,
            'atol': self.atol,
            'max_workers': self.max_workers
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({', '.join(f'{k}={v}' for k, v in self.to_dict().items())})"


# Global configuration instance
config = Config()


__all__ = ['Config', 'config']
