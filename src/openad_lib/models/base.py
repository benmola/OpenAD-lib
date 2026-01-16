"""
Base classes for all models in openad_lib.

Provides abstract base classes that define the common interface for
mechanistic and machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all AD models.
    
    This class defines the common interface that all models (mechanistic and ML)
    must implement to ensure consistent usage across the library.
    
    Attributes
    ----------
    params : Any
        Model parameters (varies by model type)
    data : pd.DataFrame, optional
        Loaded training/validation data
    results : dict, optional
        Simulation/prediction results
    metrics : dict, optional
        Evaluation metrics (RMSE, MAE, R2, etc.)
    """
    
    def __init__(self, params: Optional[Any] = None):
        """
        Initialize base model.
        
        Parameters
        ----------
        params : Any, optional
            Model-specific parameters
        """
        self.params = params
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[Dict[str, np.ndarray]] = None
        self.metrics: Optional[Dict[str, float]] = None
    
    @abstractmethod
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load training/validation data from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to data file (CSV, Excel, etc.)
            
        Returns
        -------
        data : pd.DataFrame
            Loaded dataframe
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray, optional
            True values (if None, uses self.data)
        y_pred : np.ndarray, optional
            Predicted values (if None, uses self.results)
            
        Returns
        -------
        metrics : dict
            Dictionary with metric names and values
        """
        pass
    
    def plot_results(self, **kwargs):
        """
        Visualize model results.
        
        Parameters
        ----------
        **kwargs
            Plotting options (figsize, style, etc.)
        """
        if self.results is None:
            raise ValueError("No results to plot. Run simulation/prediction first.")
        
        # Default implementation - can be overridden
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Plot first result variable
        for key, values in self.results.items():
            if isinstance(values, np.ndarray) and values.ndim == 1:
                ax.plot(values, label=key)
                break
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_metrics(self):
        """Print evaluation metrics in formatted table."""
        if self.metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        from openad_lib.utils.metrics import print_metrics
        print_metrics(self.metrics, title=f"{self.__class__.__name__} Performance")
    
    def save(self, filepath: Union[str, Path]):
        """
        Save model to file.
        
        Parameters
        ----------
        filepath : str or Path
            Output filepath (.pkl)
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """
        Load model from file.
        
        Parameters
        ----------
        filepath : str or Path
            Model filepath (.pkl)
            
        Returns
        -------
        model : BaseModel
            Loaded model instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(params={self.params})"


class MechanisticModel(BaseModel):
    """
    Base class for physics-based models (ADM1, AM2, etc.).
    
    Mechanistic models use differential equations to describe
    biological and chemical processes in anaerobic digestion.
    """
    
    @abstractmethod
    def simulate(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        t_eval: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Run mechanistic simulation.
        
        Parameters
        ----------
        t_span : tuple, optional
            Time span (t_start, t_end) for simulation
        t_eval : np.ndarray, optional
            Specific time points for evaluation
        **kwargs
            Additional simulation parameters
            
        Returns
        -------
        results : dict
            Dictionary with state trajectories and outputs
        """
        pass
    
    @abstractmethod
    def update_params(self, params: Dict[str, float]):
        """
        Update model parameters.
        
        Used during calibration to set new parameter values.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        """
        pass
    
    def get_params(self) -> Dict[str, float]:
        """
        Get current model parameters.
        
        Returns
        -------
        params : dict
            Dictionary of parameter names and values
        """
        if hasattr(self.params, '__dict__'):
            return vars(self.params)
        elif isinstance(self.params, dict):
            return self.params
        else:
            raise NotImplementedError("Custom parameter retrieval needed")


class MLModel(BaseModel):
    """
    Base class for machine learning surrogate models.
    
    ML models learn mappings from data and can provide uncertainty
    quantification for predictions.
    """
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Train the ML model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        **kwargs
            Training options (epochs, batch_size, lr, etc.)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        return_std : bool, default=False
            Whether to return uncertainty estimates
        **kwargs
            Additional prediction options
            
        Returns
        -------
        y_pred : np.ndarray
            Predictions
        y_std : np.ndarray, optional
            Uncertainty estimates (if return_std=True)
        """
        pass
    
    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Convenience method for training (alias for train).
        
        If X and y are None, uses self.data to extract features/targets.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Training features
        y : np.ndarray, optional
            Training targets
        **kwargs
            Training options
        """
        if X is None or y is None:
            if self.data is None:
                raise ValueError("No data provided. Call load_data() first or pass X, y.")
            X, y = self._extract_features_targets(self.data)
        
        return self.train(X, y, **kwargs)
    
    def _extract_features_targets(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and targets from dataframe.
        
        Override this in subclasses to customize feature extraction.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target array
        """
        raise NotImplementedError("Subclass must implement _extract_features_targets()")


__all__ = [
    'BaseModel',
    'MechanisticModel',
    'MLModel',
]
