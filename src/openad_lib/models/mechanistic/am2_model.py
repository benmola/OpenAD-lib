"""
AM2 (Anaerobic Model No. 2) Implementation.

This module provides an implementation of the simplified two-step anaerobic 
digestion model AM2, which is a reduced-order model derived from the ADM1.
The model tracks 4 state variables: S1 (COD substrate), X1 (acidogenic biomass),
S2 (VFA), X2 (methanogenic biomass), and Q (biogas production).

The AM2 model uses:
- Monod kinetics for acidogenic bacteria growth
- Haldane kinetics with pH inhibition for methanogenic bacteria growth

Example:
    >>> from openad_lib.models.mechanistic import AM2Model
    >>> model = AM2Model()
    >>> model.load_data("data.csv")
    >>> results = model.run()
    >>> model.plot_results()
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import warnings

from openad_lib.models.base import MechanisticModel


@dataclass
class AM2Parameters:
    """
    AM2 model parameters.
    
    Default parameters are calibrated values for anaerobic digestion
    of organic substrates at mesophilic conditions.
    
    Kinetic Parameters:
        m1: Maximum acidogenic biomass growth rate (d⁻¹)
        K1: Half-saturation constant for S1 (g COD/L)
        m2: Maximum methanogenic biomass growth rate (d⁻¹)
        Ki: Inhibition constant for S2 (g COD/L)
        K2: Half-saturation constant for S2 (g COD/L)
    
    Yield Coefficients:
        k1: Yield for COD degradation (g COD/g biomass)
        k2: Yield for VFA production (g VFA/g biomass)
        k3: Yield for VFA consumption (g VFA/g biomass)
        k6: Yield for CH4 production (L biogas/g biomass)
    
    Operational Parameters:
        S2in: Inlet VFA concentration (g COD/L)
        Alpha: Biomass retention factor (-)
        pHH: Upper pH limit for inhibition
        pHL: Lower pH limit for inhibition
    """
    # Kinetic parameters
    m1: float = 0.09       # d⁻¹, Maximum acidogenic biomass growth rate
    K1: float = 10.50      # g COD/L, Half-saturation constant for S1
    m2: float = 0.57       # d⁻¹, Maximum methanogenic biomass growth rate
    Ki: float = 19.93      # g COD/L, Inhibition constant for S2
    K2: float = 54.46      # g COD/L, Half-saturation constant for S2
    
    # Yield coefficients
    k1: float = 144.19     # Yield for COD degradation
    k2: float = 31.44      # Yield for VFA production
    k3: float = 535.99     # Yield for VFA consumption
    k6: float = 100.20     # Yield for CH4 production
    
    # Operational parameters
    S2in: float = 22.0     # g COD/L, Inlet VFA concentration
    Alpha: float = 0.01    # Biomass retention factor
    pHH: float = 8.5       # Upper pH limit
    pHL: float = 5.5       # Lower pH limit
    
    # Gas conversion constant
    # c = (0.001 * 8.31 * (35+273.15) / 101325) * 1000
    # At 35°C: c ≈ 0.0253

    def __post_init__(self):
        """Calculate derived parameters."""
        # Gas law constant for biogas (at 35°C, 1 atm)
        T_op = 35 + 273.15  # K
        R = 8.31  # J/(mol·K)
        P = 101325  # Pa
        self.c = (0.001 * R * T_op / P) * 1000
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            'm1': self.m1, 'K1': self.K1, 'm2': self.m2,
            'Ki': self.Ki, 'K2': self.K2, 'k1': self.k1,
            'k2': self.k2, 'k3': self.k3, 'k6': self.k6,
            'S2in': self.S2in, 'Alpha': self.Alpha,
            'pHH': self.pHH, 'pHL': self.pHL, 'c': self.c
        }


class AM2Model(MechanisticModel):
    """
    Simplified Two-Step Anaerobic Digestion Model (AM2).
    
    A reduced-order mechanistic model for anaerobic digestion that captures
    the essential dynamics of acidogenesis and methanogenesis in a 5-state
    representation.
    
    State Variables:
        S1: Organic substrate (COD) concentration (g COD/L)
        X1: Acidogenic biomass concentration (g/L)
        S2: Volatile fatty acids (VFA) concentration (g COD/L)
        X2: Methanogenic biomass concentration (g/L)
        Q: Biogas production rate (L/day or m³/day)
    
    Inputs:
        D: Dilution rate (d⁻¹)
        S1in: Inlet COD concentration (g COD/L)
        pH: Reactor pH value
    
    Attributes:
        params: AM2Parameters object containing model parameters
        results: DataFrame containing simulation results
    
    Example:
        >>> model = AM2Model()
        >>> model.load_data("data.csv")
        >>> results = model.run()
        >>> model.plot_results()
    """
    
    # State variable names
    STATE_NAMES = ["S1", "X1", "S2", "X2", "Q"]
    
    def __init__(self, params: Optional[AM2Parameters] = None):
        """
        Initialize AM2 model.
        
        Args:
            params: AM2Parameters object. If None, uses default parameters.
        """
        super().__init__(params)
        self.params = params or AM2Parameters()
        self.results: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict[str, float]] = None
        self._initial_state: np.ndarray = np.zeros(5)
    
    def load_data(
        self, 
        data_path: str,
        time_col: str = None,
        D_col: str = None,
        S1in_col: str = None,
        pH_col: str = None,
        S1out_col: str = None,
        S2out_col: str = None,
        Q_col: str = None
    ) -> None:
        """
        Load input data from file.
        
        The data file should contain columns for time, dilution rate (D),
        inlet substrate concentration (S1in), and pH. Optionally, measured
        outputs (S1out, S2out, Q) can be included for comparison.
        
        Args:
            data_path: Path to CSV file with input data
            time_col: Column name for time (default: column 0)
            D_col: Column name for dilution rate (default: column 1)
            S1in_col: Column name for inlet S1 (default: column 2)
            pH_col: Column name for pH (default: column 4)
            S1out_col: Column name for measured S1 output (default: column 5)
            S2out_col: Column name for measured S2 output (default: column 6)
            Q_col: Column name for measured biogas (default: column 7)
        """
        # Load raw data
        data = np.genfromtxt(data_path, delimiter=',', skip_header=1, filling_values=np.nan)
        
        # Create DataFrame with standard column names
        self.data = pd.DataFrame({
            'time': data[:, 0],
            'D': data[:, 1],
            'S1in': data[:, 2],
            'OLR': data[:, 3] if data.shape[1] > 3 else data[:, 2] * data[:, 1],
            'pH': data[:, 4] if data.shape[1] > 4 else np.ones(len(data)) * 7.0,
            'S1out': data[:, 5] if data.shape[1] > 5 else np.nan * np.ones(len(data)),
            'S2out': data[:, 6] if data.shape[1] > 6 else np.nan * np.ones(len(data)),
            'Q': data[:, 7] if data.shape[1] > 7 else np.nan * np.ones(len(data))
        })
        
        # Set initial state from measured data if available
        if not np.isnan(self.data['S1out'].iloc[0]):
            self._initial_state[0] = self.data['S1out'].iloc[0]  # S1
        else:
            self._initial_state[0] = 10.0  # Default initial S1
            
        self._initial_state[1] = 0.2  # X1 (initial acidogenic biomass)
        
        if not np.isnan(self.data['S2out'].iloc[0]):
            self._initial_state[2] = self.data['S2out'].iloc[0]  # S2
        else:
            self._initial_state[2] = 30.0  # Default initial S2
            
        self._initial_state[3] = 0.6  # X2 (initial methanogenic biomass)
        
        if not np.isnan(self.data['Q'].iloc[0]):
            self._initial_state[4] = self.data['Q'].iloc[0]  # Q
        else:
            self._initial_state[4] = 0.0  # Default initial Q
    
    def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = 'time',
        D_col: str = 'D',
        S1in_col: str = 'S1in',
        pH_col: str = 'pH',
        S1out_col: str = 'S1out',
        S2out_col: str = 'S2out',
        Q_col: str = 'Q'
    ) -> None:
        """
        Load input data from a pandas DataFrame.
        
        Args:
            df: DataFrame with input data
            time_col: Column name for time
            D_col: Column name for dilution rate
            S1in_col: Column name for inlet substrate concentration
            pH_col: Column name for pH
            S1out_col: Column name for measured S1 output (optional)
            S2out_col: Column name for measured S2 output (optional)
            Q_col: Column name for measured biogas (optional)
        """
        self.data = pd.DataFrame({
            'time': df[time_col].values,
            'D': df[D_col].values,
            'S1in': df[S1in_col].values,
            'pH': df[pH_col].values if pH_col in df.columns else np.ones(len(df)) * 7.0,
            'S1out': df[S1out_col].values if S1out_col in df.columns else np.nan * np.ones(len(df)),
            'S2out': df[S2out_col].values if S2out_col in df.columns else np.nan * np.ones(len(df)),
            'Q': df[Q_col].values if Q_col in df.columns else np.nan * np.ones(len(df))
        })
        
        # Set initial state from measured data if available
        if S1out_col in df.columns and not np.isnan(df[S1out_col].iloc[0]):
            self._initial_state[0] = df[S1out_col].iloc[0]
        else:
            self._initial_state[0] = 10.0
            
        self._initial_state[1] = 0.2  # X1
        
        if S2out_col in df.columns and not np.isnan(df[S2out_col].iloc[0]):
            self._initial_state[2] = df[S2out_col].iloc[0]
        else:
            self._initial_state[2] = 30.0
            
        self._initial_state[3] = 0.6  # X2
        
        if Q_col in df.columns and not np.isnan(df[Q_col].iloc[0]):
            self._initial_state[4] = df[Q_col].iloc[0]
        else:
            self._initial_state[4] = 0.0
    
    def set_initial_conditions(
        self,
        S1: float = 10.0,
        X1: float = 0.2,
        S2: float = 30.0,
        X2: float = 0.6,
        Q: float = 0.0
    ) -> None:
        """
        Manually set initial conditions for simulation.
        
        Args:
            S1: Initial COD concentration (g COD/L)
            X1: Initial acidogenic biomass (g/L)
            S2: Initial VFA concentration (g COD/L)
            X2: Initial methanogenic biomass (g/L)
            Q: Initial biogas production rate
        """
        self._initial_state = np.array([S1, X1, S2, X2, Q])
    
    def _am2_ode(
        self, 
        x: np.ndarray, 
        t: float, 
        D: float, 
        S1in: float, 
        pH: float
    ) -> List[float]:
        """
        AM2 differential equations.
        
        Args:
            x: State vector [S1, X1, S2, X2, Q]
            t: Current time
            D: Dilution rate (d⁻¹)
            S1in: Inlet substrate concentration (g COD/L)
            pH: Reactor pH
        
        Returns:
            List of derivatives [dS1, dX1, dS2, dX2, Q]
        """
        p = self.params
        
        # Unpack states
        S1, X1, S2, X2, Q_state = x
        
        # Monod kinetics for acidogenic growth
        MU1 = p.m1 * (S1 / (S1 + p.K1))
        
        # Haldane kinetics with pH inhibition for methanogenic growth
        # Base Haldane kinetics
        MU2_base = p.m2 * (S2 / ((S2 * S2) / p.Ki + S2 + p.K2))
        # pH inhibition factor
        pH_factor = np.exp(-4 * ((pH - p.pHH) / (p.pHH - p.pHL)) ** 2)
        MU2 = MU2_base - (MU2_base * pH_factor)
        
        # Mass balances
        dS1 = D * (S1in - S1) - p.k1 * MU1 * X1
        dX1 = (MU1 - p.Alpha * D) * X1
        dS2 = D * (p.S2in - S2) + p.k2 * MU1 * X1 - p.k3 * MU2 * X2
        dX2 = (MU2 - p.Alpha * D) * X2
        Q = p.k6 * MU2 * X2 * p.c
        
        return [dS1, dX1, dS2, dX2, Q]
    
    def _simulate(self, params: Optional[AM2Parameters] = None) -> np.ndarray:
        """
        Internal simulation function.
        
        Args:
            params: Parameters to use (defaults to self.params)
        
        Returns:
            Array of shape (n_steps, 5) with state trajectories
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if params is None:
            params = self.params
        
        t = self.data['time'].values
        D = self.data['D'].values
        S1in = self.data['S1in'].values
        pH = self.data['pH'].values
        
        # Initialize results array
        AM2 = np.zeros((len(t), 5))
        AM2[0] = self._initial_state.copy()
        AM20 = AM2[0].copy()
        
        # Integrate step by step
        for i in range(len(t) - 1):
            ts = [t[i], t[i + 1]]
            y = odeint(
                self._am2_ode, AM20, ts,
                args=(D[i], S1in[i], pH[i])
            )
            AM20 = y[-1]
            AM2[i + 1] = AM20
        
        return AM2
    
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run AM2 simulation.
        
        Args:
            verbose: Whether to print progress information
        
        Returns:
            DataFrame with simulation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if verbose:
            print("Running AM2 simulation...")
        
        # Run simulation
        AM2 = self._simulate()
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'time': self.data['time'],
            'S1': AM2[:, 0],
            'X1': AM2[:, 1],
            'S2': AM2[:, 2],
            'X2': AM2[:, 3],
            'Q': AM2[:, 4]
        })
        
        # Add input data to results
        self.results['D'] = self.data['D']
        self.results['S1in'] = self.data['S1in']
        self.results['pH'] = self.data['pH']
        
        # Add measured data if available
        if 'S1out' in self.data.columns:
            self.results['S1_measured'] = self.data['S1out']
        if 'S2out' in self.data.columns:
            self.results['S2_measured'] = self.data['S2out']
        if 'Q' in self.data.columns:
            self.results['Q_measured'] = self.data['Q']
        
        # Calculate growth rates
        self.results['MU1'] = self.params.m1 * (AM2[:, 0] / (AM2[:, 0] + self.params.K1))
        
        # Haldane with pH inhibition
        S2 = AM2[:, 2]
        pH_vals = self.data['pH'].values
        MU2_base = self.params.m2 * (S2 / ((S2 * S2) / self.params.Ki + S2 + self.params.K2))
        pH_factor = np.exp(-4 * ((pH_vals - self.params.pHH) / (self.params.pHH - self.params.pHL)) ** 2)
        self.results['MU2'] = MU2_base - (MU2_base * pH_factor)
        
        if verbose:
            print("Simulation complete!")
        
        return self.results
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate evaluation metrics comparing simulation to measured data.
        
        Returns:
            Dictionary with RMSE, MAE, and R² for each output variable
        """
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        
        from sklearn.metrics import mean_absolute_error, r2_score
        
        def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
            return np.sqrt(((predictions - targets) ** 2).mean())
        
        metrics = {}
        
        # S1 (COD)
        if 'S1_measured' in self.results.columns:
            S1_pred = self.results['S1'].values
            S1_meas = self.results['S1_measured'].values
            mask = ~np.isnan(S1_meas)
            if mask.sum() > 0:
                metrics['S1'] = {
                    'RMSE': rmse(S1_pred[mask], S1_meas[mask]),
                    'MAE': mean_absolute_error(S1_meas[mask], S1_pred[mask]),
                    'R2': r2_score(S1_meas[mask], S1_pred[mask])
                }
        
        # S2 (VFA)
        if 'S2_measured' in self.results.columns:
            S2_pred = self.results['S2'].values
            S2_meas = self.results['S2_measured'].values
            mask = ~np.isnan(S2_meas)
            if mask.sum() > 0:
                metrics['S2'] = {
                    'RMSE': rmse(S2_pred[mask], S2_meas[mask]),
                    'MAE': mean_absolute_error(S2_meas[mask], S2_pred[mask]),
                    'R2': r2_score(S2_meas[mask], S2_pred[mask])
                }
        
        # Q (Biogas)
        if 'Q_measured' in self.results.columns:
            Q_pred = self.results['Q'].values
            Q_meas = self.results['Q_measured'].values
            mask = ~np.isnan(Q_meas)
            if mask.sum() > 0:
                metrics['Q'] = {
                    'RMSE': rmse(Q_pred[mask], Q_meas[mask]),
                    'MAE': mean_absolute_error(Q_meas[mask], Q_pred[mask]),
                    'R2': r2_score(Q_meas[mask], Q_pred[mask])
                }
        
        return metrics
    
    def print_metrics(self) -> None:
        """Print formatted evaluation metrics."""
        metrics = self.evaluate()
        
        print('=' * 60)
        print('AM2 Model Evaluation Metrics')
        print('=' * 60)
        
        for var, m in metrics.items():
            print(f"\n{var}:")
            print(f"  RMSE: {m['RMSE']:.4f}")
            print(f"  MAE:  {m['MAE']:.4f}")
            print(f"  R²:   {m['R2']:.4f}")
    
    def print_parameters(self) -> None:
        """Print current model parameters."""
        p = self.params
        print('=' * 60)
        print('AM2 Model Parameters')
        print('=' * 60)
        print(f"µ1max (m1): {p.m1:.4f} d⁻¹  (Max acidogenic growth rate)")
        print(f"K1:         {p.K1:.4f} g COD/L  (Half-saturation for S1)")
        print(f"µ2max (m2): {p.m2:.4f} d⁻¹  (Max methanogenic growth rate)")
        print(f"Ki:         {p.Ki:.4f} g COD/L  (Inhibition constant)")
        print(f"K2:         {p.K2:.4f} g COD/L  (Half-saturation for S2)")
        print(f"k1:         {p.k1:.4f}  (COD degradation yield)")
        print(f"k2:         {p.k2:.4f}  (VFA production yield)")
        print(f"k3:         {p.k3:.4f}  (VFA consumption yield)")
        print(f"k6:         {p.k6:.4f}  (CH4 production yield)")
    
    def get_results(self) -> pd.DataFrame:
        """Return results DataFrame."""
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        return self.results
    
    def plot_results(
        self,
        variables: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 12),
        show_measured: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot simulation results.
        
        Args:
            variables: List of variables to plot. If None, plots S1, S2, Q.
            figsize: Figure size (width, height)
            show_measured: Whether to show measured data points
            save_path: Path to save the plot. If None, saves to images/am2_simulation.png
        """
        import matplotlib.pyplot as plt
        import os
        
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        
        if variables is None:
            variables = ['S1', 'S2', 'Q']
        
        # Apply consistent styling
        plt.style.use('bmh')
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
        
        if n_vars == 1:
            axes = [axes]
        
        time = self.results['time']
        
        output_labels = {
            'S1': 'COD (S1) [g COD/L]',
            'S2': 'VFA (S2) [g COD/L]',
            'Q': 'Biogas Production [L/day]',
            'X1': 'Acidogenic Biomass (X1) [g/L]',
            'X2': 'Methanogenic Biomass (X2) [g/L]',
            'MU1': 'Acidogenic Growth Rate (µ1) [d⁻¹]',
            'MU2': 'Methanogenic Growth Rate (µ2) [d⁻¹]'
        }
        
        measured_cols = {
            'S1': 'S1_measured',
            'S2': 'S2_measured',
            'Q': 'Q_measured'
        }
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            # Plot simulated data
            ax.plot(
                time, self.results[var], 
                color='#E67E22', linestyle='-', linewidth=2,
                label=f'Predicted {var}'
            )
            
            # Plot measured data if available and requested
            meas_col = measured_cols.get(var)
            if show_measured and meas_col and meas_col in self.results.columns:
                meas_data = self.results[meas_col]
                if not meas_data.isna().all():
                    ax.plot(
                        time, meas_data, 
                        'o', color='#2E86C1', markersize=6,
                        label=f'Measured {var}', alpha=0.7
                    )
            
            # Apply consistent formatting
            ax.set_xlabel('Time (days)' if i == n_vars-1 else '', fontsize=14, fontweight='bold')
            ax.set_ylabel(output_labels.get(var, var), fontsize=14, fontweight='bold')
            ax.set_title(f'{output_labels.get(var, var)} over Time', fontsize=16, pad=20)
            ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            # Default: save to images folder in project root
            # Try to find project root (3 levels up from this file)
            try:
                current_file = os.path.abspath(__file__)
                # Go up 5 levels: am2_model.py -> mechanistic -> models -> openad_lib -> src -> project_root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
                images_dir = os.path.join(project_root, 'images')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                save_path = os.path.join(images_dir, 'am2_simulation.png')
            except:
                save_path = 'am2_simulation.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        # plt.show()  # Commented out for non-interactive use
    
    def save_results(self, output_path: str) -> None:
        """
        Save simulation results to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        self.results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    # ========== Base Class Interface Methods ==========
    
    def simulate(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        t_eval: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Run mechanistic simulation (implements MechanisticModel.simulate).
        
        This is the base class interface. Use run() for the full workflow.
        
        Parameters
        ----------
        t_span : tuple, optional
            Not used for AM2 (uses loaded data time points)
        t_eval : np.ndarray, optional
            Not used for AM2 (uses loaded data time points)
        **kwargs
            Additional parameters
            
        Returns
        -------
        results : dict
            Dictionary with state trajectories
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Run simulation
        AM2 = self._simulate()
        
        # Return as dictionary
        return {
            'time': self.data['time'].values,
            'S1': AM2[:, 0],
            'X1': AM2[:, 1],
            'S2': AM2[:, 2],
            'X2': AM2[:, 3],
            'Q': AM2[:, 4]
        }
    
    def update_params(self, params: Dict[str, float]):
        """
        Update model parameters (implements MechanisticModel.update_params).
        
        Used during calibration to set new parameter values.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        """
        for name, value in params.items():
            if hasattr(self.params, name):
                setattr(self.params, name, value)
            else:
                raise ValueError(f"Unknown parameter: {name}")
    
    def evaluate(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (overrides base class).
        
        If y_true and y_pred are provided, computes metrics directly.
        Otherwise, uses the existing evaluate() logic for AM2.
        
        Parameters
        ----------
        y_true : np.ndarray, optional
            True values
        y_pred : np.ndarray, optional
            Predicted values
            
        Returns
        -------
        metrics : dict
            Dictionary with metric names and values
        """
        if y_true is not None and y_pred is not None:
            from openad_lib.utils.metrics import compute_metrics
            self.metrics = compute_metrics(y_true, y_pred)
            return self.metrics
        
        # Use existing AM2-specific evaluate logic
        if self.results is None:
            raise ValueError("No results to evaluate. Run simulation first.")
        
        def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
            return np.sqrt(np.mean((predictions - targets) ** 2))
        
        metrics = {}
        
        # Calculate RMSE for each measured output variable
        if 'S1_measured' in self.results.columns:
            valid_mask = ~self.results['S1_measured'].isna()
            if valid_mask.sum() > 0:
                metrics['S1_RMSE'] = rmse(
                    self.results.loc[valid_mask, 'S1'].values,
                    self.results.loc[valid_mask, 'S1_measured'].values
                )
        
        if 'S2_measured' in self.results.columns:
            valid_mask = ~self.results['S2_measured'].isna()
            if valid_mask.sum() > 0:
                metrics['S2_RMSE'] = rmse(
                    self.results.loc[valid_mask, 'S2'].values,
                    self.results.loc[valid_mask, 'S2_measured'].values
                )
        
        if 'Q_measured' in self.results.columns:
            valid_mask = ~self.results['Q_measured'].isna()
            if valid_mask.sum() > 0:
                metrics['Q_RMSE'] = rmse(
                    self.results.loc[valid_mask, 'Q'].values,
                    self.results.loc[valid_mask, 'Q_measured'].values
                )
        
        if metrics:
            metrics['Overall_RMSE'] = np.mean(list(metrics.values()))
        
        self.metrics = metrics
        return metrics

