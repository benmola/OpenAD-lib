"""
AM2 Model Calibration using Optuna.

This module provides a framework for calibrating AM2 model parameters
to fit experimental data using Bayesian optimization via Optuna.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    optuna = None

from openad_lib.models.mechanistic.am2_model import AM2Model, AM2Parameters


class AM2Calibrator:
    """
    Calibrator for AM2 model parameters using Optuna.
    
    Attributes:
        model: AM2Model instance to calibrate
        data: DataFrame containing experimental data
        study: Optuna study object (available after run_optimization)
    """
    
    def __init__(self, model: Optional[AM2Model] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the calibrator.
        
        Args:
            model: AM2Model instance. If None, creates a new one.
            data: DataFrame with experimental data (time, S1, S2, Q).
                  If None, expected to be loaded into model.
        """
        if optuna is None:
            raise ImportError("Optuna is required for calibration. Install with: pip install optuna")
            
        self.model = model or AM2Model()
        if data is not None:
            self.model.load_data_from_dataframe(
                data, 
                S1out_col='S1_measured' if 'S1_measured' in data.columns else 'S1out',
                S2out_col='S2_measured' if 'S2_measured' in data.columns else 'S2out',
                Q_col='Q_measured' if 'Q_measured' in data.columns else 'Q'
            )
        
        self.study: Optional[optuna.Study] = None
        
        # Default parameter ranges (min, max)
        # Based on typical variations around the default values
        self.param_ranges = {
            'm1': (0.01, 1.0),      # Default: 0.09
            'K1': (1.0, 50.0),      # Default: 10.50
            'm2': (0.1, 2.0),       # Default: 0.57
            'Ki': (1.0, 100.0),     # Default: 19.93
            'K2': (1.0, 100.0),     # Default: 54.46
            'k1': (50.0, 200.0),    # Default: 144.19
            'k2': (10.0, 100.0),    # Default: 31.44
            'k3': (100.0, 1000.0),  # Default: 535.99
            'k6': (50.0, 200.0)     # Default: 100.20
        }

    def objective(
        self, 
        trial: Trial, 
        params_to_tune: List[str],
        weights: Dict[str, float]
    ) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
            params_to_tune: List of parameter names to tune
            weights: Dictionary of weights for each output variable (S1, S2, Q)
            
        Returns:
            Weighted Sum of Squared Errors (SSE)
        """
        # Suggest parameters
        current_params = {}
        for param in params_to_tune:
            low, high = self.param_ranges.get(param, (0.01, 100.0))
            current_params[param] = trial.suggest_float(param, low, high)
        
        # Update model parameters
        original_params = self.model.params.to_dict()
        for k, v in current_params.items():
            setattr(self.model.params, k, v)
        
        try:
            # Run simulation
            results = self.model.run(verbose=False)
            
            # Calculate error
            total_error = 0.0
            
            # S1 (COD) Error
            if weights.get('S1', 0) > 0 and 'S1_measured' in results.columns:
                target = results['S1_measured'].values
                pred = results['S1'].values
                mask = ~np.isnan(target)
                if mask.sum() > 0:
                    # Normalized SSE
                    mse = np.mean((target[mask] - pred[mask])**2)
                    var_scale = np.var(target[mask]) + 1e-6
                    total_error += weights['S1'] * (mse / var_scale)

            # S2 (VFA) Error
            if weights.get('S2', 0) > 0 and 'S2_measured' in results.columns:
                target = results['S2_measured'].values
                pred = results['S2'].values
                mask = ~np.isnan(target)
                if mask.sum() > 0:
                    mse = np.mean((target[mask] - pred[mask])**2)
                    var_scale = np.var(target[mask]) + 1e-6
                    total_error += weights['S2'] * (mse / var_scale)

            # Q (Biogas) Error
            if weights.get('Q', 0) > 0 and 'Q_measured' in results.columns:
                target = results['Q_measured'].values
                pred = results['Q'].values
                mask = ~np.isnan(target)
                if mask.sum() > 0:
                    mse = np.mean((target[mask] - pred[mask])**2)
                    var_scale = np.var(target[mask]) + 1e-6
                    total_error += weights['Q'] * (mse / var_scale)
            
            return total_error
            
        except Exception as e:
            # If simulation fails (e.g. unstable), return infinity
            return float('inf')
        finally:
            # Restore original parameters (optional, but good practice if using same model instance)
            # Actually for speed in optimization, we don't restore every time, 
            # we just overwrite in next trial. 
            pass

    def calibrate(
        self,
        params_to_tune: Optional[List[str]] = None,
        n_trials: int = 100,
        weights: Optional[Dict[str, float]] = None,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Run the calibration process.
        
        Args:
            params_to_tune: List of parameter names to optimize. 
                            If None, optimizes kinetic parameters (m1, K1, m2, Ki, K2).
            n_trials: Number of optimization trials
            weights: Weights for error components {'S1': 1.0, 'S2': 1.0, 'Q': 1.0}
            timeout: Optimization timeout in seconds
            show_progress_bar: Whether to show Optuna progress bar
            
        Returns:
            Dictionary with best parameters and optimization study
        """
        if self.model.data is None:
            raise ValueError("No data loaded in model. Cannot calibrate.")
            
        if params_to_tune is None:
            params_to_tune = ['m1', 'K1', 'm2', 'Ki', 'K2']
            
        if weights is None:
            weights = {'S1': 1.0, 'S2': 1.0, 'Q': 1.0}
            
        # Create study
        self.study = optuna.create_study(direction="minimize")
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, params_to_tune, weights),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar
        )
        
        best_params = self.study.best_params
        
        # Apply best parameters to model
        for k, v in best_params.items():
            setattr(self.model.params, k, v)
            
        print("Calibration complete.")
        print(f"Best trial value (Weighted Normalized MSE): {self.study.best_value:.4f}")
        print("Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v:.4f}")
            
        return best_params

    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("Run calibrate() first.")
        
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            print("Plotly is required for Optuna visualization.")

    def plot_param_importances(self):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("Run calibrate() first.")
        
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            fig.show()
        except ImportError:
            print("Plotly is required for Optuna visualization.")
