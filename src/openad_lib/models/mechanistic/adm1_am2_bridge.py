"""
ADM1-AM2 Model Hierarchy Bridge
================================

This module provides systematic aggregation of ADM1 states (32 variables) into 
AM2 aggregated variables (4 variables) following biochemical lumping principles.

The aggregation preserves mass and energy balance through stoichiometric factors
as described in Bernard et al. [16].

References:
    [16] Bernard, O., Hadj-Sadok, Z., Dochain, D., Genovesi, A., & Steyer, J. P. (2001).
         Dynamical model development and parameter identification for an anaerobic 
         wastewater treatment process. Biotechnology and bioengineering, 75(4), 424-438.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional


# Stoichiometric factors for VFA molecular weights (g/mol)
VFA_MW = {
    'va': 208.0,   # Valerate (C5H10O2)
    'bu': 160.0,   # Butyrate (C4H8O2)
    'pro': 112.0,  # Propionate (C3H6O2)
    'ac': 64.0     # Acetate (C2H4O2)
}

# Biomass yield factor (COD basis)
BIOMASS_YIELD = 1.55


def aggregate_adm1_to_am2(
    adm1_states: Union[pd.DataFrame, Dict[str, np.ndarray]],
    time: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Aggregate ADM1 states to AM2 variables using biochemical lumping.
    
    Aggregation equations:
        S̃₁ = S_su + S_aa + S_lcfa + X_c + X_ch + X_pr + X_li
        S̃₂ = 1000 * (S_va/208 + S_bu/160 + S_pro/112 + S_ac/64)
        X̃₁ = (X_su + X_aa + X_lcfa + X_c4 + X_pro) / 1.55
        X̃₂ = (X_ac + X_h2) / 1.55
    
    Parameters
    ----------
    adm1_states : pd.DataFrame or dict
        ADM1 state trajectories. If DataFrame, columns should be state names.
        If dict, keys are state names and values are arrays.
    time : np.ndarray, optional
        Time vector. If None and adm1_states is DataFrame, uses index.
    
    Returns
    -------
    pd.DataFrame
        AM2 aggregated variables with columns ['time', 'S1', 'S2', 'X1', 'X2']
    
    Examples
    --------
    >>> # From ADM1 simulation results
    >>> adm1_results = adm1_model.simulate(influent_df)
    >>> am2_data = aggregate_adm1_to_am2(adm1_results['states'])
    
    >>> # Use for AM2 calibration
    >>> am2_model.load_from_dataframe(am2_data, ...)
    """
    # Convert to DataFrame if dict
    if isinstance(adm1_states, dict):
        df = pd.DataFrame(adm1_states)
    else:
        df = adm1_states.copy()
    
    # Extract time vector
    if time is None:
        if 'time' in df.columns:
            time = df['time'].values
        else:
            time = df.index.values
    
    # Aggregate S̃₁: Organic substrates and particulates
    # S̃₁ = S_su + S_aa + S_fa + X_xc + X_ch + X_pr + X_li
    S1_components = ['S_su', 'S_aa', 'S_fa', 'X_xc', 'X_ch', 'X_pr', 'X_li']
    S1 = sum(df[comp].values for comp in S1_components if comp in df.columns)
    
    # Aggregate S̃₂: VFAs (molar basis with molecular weight normalization)
    # S̃₂ = 1000 * (S_va/208 + S_bu/160 + S_pro/112 + S_ac/64)
    S2 = 1000.0 * (
        df['S_va'].values / VFA_MW['va'] +
        df['S_bu'].values / VFA_MW['bu'] +
        df['S_pro'].values / VFA_MW['pro'] +
        df['S_ac'].values / VFA_MW['ac']
    )
    
    # Aggregate X̃₁: Acidogenic biomass (yield-corrected)
    # X̃₁ = (X_su + X_aa + X_fa + X_c4 + X_pro) / 1.55
    X1_components = ['X_su', 'X_aa', 'X_fa', 'X_c4', 'X_pro']
    X1 = sum(df[comp].values for comp in X1_components if comp in df.columns) / BIOMASS_YIELD
    
    # Aggregate X̃₂: Methanogenic biomass (yield-corrected)
    # X̃₂ = (X_ac + X_h2) / 1.55
    X2_components = ['X_ac', 'X_h2']
    X2 = sum(df[comp].values for comp in X2_components if comp in df.columns) / BIOMASS_YIELD
    
    # Create AM2 dataframe
    am2_df = pd.DataFrame({
        'time': time,
        'S1': S1,
        'S2': S2,
        'X1': X1,
        'X2': X2
    })
    
    return am2_df


def calculate_am2_influent(
    influent_data: pd.DataFrame,
    time: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Calculate AM2 influent variables (S1in, S2in) from ADM1 influent characterization.
    
    Uses the same aggregation equations as aggregate_adm1_to_am2:
        S1in = S_su_in + S_aa_in + S_fa_in + X_c_in + X_ch_in + X_pr_in + X_li_in
        S2in = 1000 * (S_va_in/208 + S_bu_in/160 + S_pro_in/112 + S_ac_in/64)
    
    Parameters
    ----------
    influent_data : pd.DataFrame
        ADM1 influent characterization data with columns for influent concentrations.
        Expected columns: S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, 
                         S_ac_in, X_c_in, X_ch_in, X_pr_in, X_li_in
    time : np.ndarray, optional
        Time vector. If None, uses index from influent_data.
    
    Returns
    -------
    pd.DataFrame
        AM2 influent variables with columns ['time', 'S1in', 'S2in']
    
    Examples
    --------
    >>> # From feedstock characterization
    >>> influent_df = openad.acod.generate_influent_data(feedstock_data)
    >>> am2_influent = calculate_am2_influent(influent_df)
    >>> print(am2_influent[['time', 'S1in', 'S2in']])
    """
    df = influent_data.copy()
    
    # Extract time vector
    if time is None:
        if 'time' in df.columns:
            time = df['time'].values
        else:
            time = df.index.values
    
    # Aggregate S1in: Organic substrates and particulates in influent
    # S1in = S_su_in + S_aa_in + S_fa_in + X_c_in + X_ch_in + X_pr_in + X_li_in
    S1in_components = ['S_su_in', 'S_aa_in', 'S_fa_in', 'X_c_in', 'X_ch_in', 'X_pr_in', 'X_li_in']
    S1in = sum(df[comp].values for comp in S1in_components if comp in df.columns)
    
    # Aggregate S2in: VFAs in influent (molar basis with molecular weight normalization)
    # S2in = 1000 * (S_va_in/208 + S_bu_in/160 + S_pro_in/112 + S_ac_in/64)
    S2in_components = {
        'S_va_in': VFA_MW['va'],
        'S_bu_in': VFA_MW['bu'],
        'S_pro_in': VFA_MW['pro'],
        'S_ac_in': VFA_MW['ac']
    }
    
    S2in = 1000.0 * sum(
        df[comp].values / mw 
        for comp, mw in S2in_components.items() 
        if comp in df.columns
    )
    
    # Create AM2 influent dataframe
    am2_influent_df = pd.DataFrame({
        'time': time,
        'S1in': S1in,
        'S2in': S2in
    })
    
    return am2_influent_df


def validate_aggregation(
    adm1_states: pd.DataFrame,
    am2_states: pd.DataFrame,
    tolerance: float = 1e-6
) -> Dict[str, bool]:
    """
    Validate mass and energy balance preservation in aggregation.
    
    Parameters
    ----------
    adm1_states : pd.DataFrame
        Original ADM1 states
    am2_states : pd.DataFrame
        Aggregated AM2 states
    tolerance : float
        Numerical tolerance for validation
    
    Returns
    -------
    dict
        Validation results for each conservation law
    """
    results = {}
    
    # Check total COD conservation
    # Total COD should be approximately preserved
    adm1_total_cod = (
        adm1_states[['S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac']].sum(axis=1) +
        adm1_states[['X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa', 
                     'X_c4', 'X_pro', 'X_ac', 'X_h2']].sum(axis=1)
    )
    
    am2_total_cod = am2_states['S1'] + am2_states['S2'] + am2_states['X1'] + am2_states['X2']
    
    cod_error = np.abs(adm1_total_cod.values - am2_total_cod.values).max()
    results['cod_conservation'] = cod_error < tolerance
    results['cod_max_error'] = cod_error
    
    # Check non-negativity
    results['s1_positive'] = (am2_states['S1'] >= 0).all()
    results['s2_positive'] = (am2_states['S2'] >= 0).all()
    results['x1_positive'] = (am2_states['X1'] >= 0).all()
    results['x2_positive'] = (am2_states['X2'] >= 0).all()
    
    return results


def get_required_adm1_states() -> Dict[str, list]:
    """
    Get the ADM1 states required for each AM2 variable.
    
    Returns
    -------
    dict
        Mapping of AM2 variables to required ADM1 states
    """
    return {
        'S1': ['S_su', 'S_aa', 'S_fa', 'X_xc', 'X_ch', 'X_pr', 'X_li'],
        'S2': ['S_va', 'S_bu', 'S_pro', 'S_ac'],
        'X1': ['X_su', 'X_aa', 'X_fa', 'X_c4', 'X_pro'],
        'X2': ['X_ac', 'X_h2']
    }


def compute_aggregation_info_loss(
    adm1_states: pd.DataFrame,
    am2_states: pd.DataFrame
) -> Dict[str, float]:
    """
    Quantify information loss from ADM1 to AM2 aggregation.
    
    Parameters
    ----------
    adm1_states : pd.DataFrame
        Original ADM1 states (32 variables)
    am2_states : pd.DataFrame
        Aggregated AM2 states (4 variables)
    
    Returns
    -------
    dict
        Information loss metrics
    """
    metrics = {}
    
    # Dimensionality reduction
    n_adm1 = len([c for c in adm1_states.columns if c != 'time'])
    n_am2 = len([c for c in am2_states.columns if c != 'time'])
    metrics['dimension_reduction_ratio'] = n_am2 / n_adm1
    
    # Variance preservation (for each AM2 variable)
    required_states = get_required_adm1_states()
    
    for am2_var, adm1_vars in required_states.items():
        # Variance in ADM1 components
        adm1_var = sum(adm1_states[v].var() for v in adm1_vars if v in adm1_states.columns)
        # Variance in aggregated AM2 variable
        am2_var = am2_states[am2_var].var()
        # Variance preservation ratio
        metrics[f'{am2_var}_variance_ratio'] = am2_var / adm1_var if adm1_var > 0 else 0
    
    return metrics
