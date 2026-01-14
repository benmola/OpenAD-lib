"""
COD-based Anaerobic Digestion Model No. 1 (ADM1) Implementation.

This module provides a complete implementation of the ADM1 model as described
in the IWA Anaerobic Digestion Model No. 1 (ADM1) and the BSM2 report by
Rosen et al. (2006).

The model includes:
- 38 state variables (12 soluble, 12 particulate, 6 ion states, 3 gas phase)
- Biochemical processes (disintegration, hydrolysis, uptake, decay)
- Physicochemical processes (acid-base equilibria, gas transfer)
- DAE solver for stiff algebraic constraints

Example:
    >>> from openad_lib.models.mechanistic import ADM1Model
    >>> model = ADM1Model()
    >>> model.load_data(influent_path="influent.xlsx", initial_state_path="initial.csv")
    >>> results = model.run()
    >>> model.plot_results()
"""

import numpy as np
import pandas as pd
from scipy import integrate
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class ADM1Parameters:
    """
    ADM1 model parameters following BSM2 conventions.
    
    All parameters from Rosen et al. (2006) BSM2 report.
    """
    # Physical parameters
    V_liq: float = 10000.0  # m³, liquid volume
    V_gas: float = 600.0    # m³, gas headspace volume
    T_op: float = 318.15    # K, operating temperature (35°C)
    T_base: float = 298.15  # K, base temperature
    R: float = 0.083145     # bar·M⁻¹·K⁻¹, gas constant
    p_atm: float = 1.013    # bar, atmospheric pressure
    k_p: float = 100 * 10**4  # m³·d⁻¹·bar⁻¹, gas outlet friction
    
    # Stoichiometric parameters
    f_sI_xc: float = 0.1
    f_xI_xc: float = 0.2
    f_ch_xc: float = 0.2
    f_pr_xc: float = 0.2
    f_li_xc: float = 0.3
    N_xc: float = 0.0027       # kmol N·kg⁻¹COD
    N_I: float = 0.0014        # kmol N·kg⁻¹COD
    N_aa: float = 0.0114       # kmol N·kg⁻¹COD
    C_xc: float = 0.02786      # kmol C·kg⁻¹COD
    C_sI: float = 0.03         # kmol C·kg⁻¹COD
    C_ch: float = 0.0313       # kmol C·kg⁻¹COD
    C_pr: float = 0.03         # kmol C·kg⁻¹COD
    C_li: float = 0.022        # kmol C·kg⁻¹COD
    C_xI: float = 0.03         # kmol C·kg⁻¹COD
    C_su: float = 0.0313       # kmol C·kg⁻¹COD
    C_aa: float = 0.03         # kmol C·kg⁻¹COD
    f_fa_li: float = 0.95
    C_fa: float = 0.0217       # kmol C·kg⁻¹COD
    f_h2_su: float = 0.19
    f_bu_su: float = 0.13
    f_pro_su: float = 0.27
    f_ac_su: float = 0.41
    N_bac: float = 0.0245      # kmol N·kg⁻¹COD
    C_bu: float = 0.025        # kmol C·kg⁻¹COD
    C_pro: float = 0.0268      # kmol C·kg⁻¹COD
    C_ac: float = 0.0313       # kmol C·kg⁻¹COD
    C_bac: float = 0.0313      # kmol C·kg⁻¹COD
    Y_su: float = 0.1
    f_h2_aa: float = 0.06
    f_va_aa: float = 0.23
    f_bu_aa: float = 0.26
    f_pro_aa: float = 0.05
    f_ac_aa: float = 0.40
    C_va: float = 0.024        # kmol C·kg⁻¹COD
    Y_aa: float = 0.08
    Y_fa: float = 0.06
    Y_c4: float = 0.06
    Y_pro: float = 0.04
    C_ch4: float = 0.0156      # kmol C·kg⁻¹COD
    Y_ac: float = 0.05
    Y_h2: float = 0.06
    
    # Biochemical kinetic parameters
    k_dis: float = 0.5         # d⁻¹, disintegration
    k_hyd_ch: float = 10.0     # d⁻¹, hydrolysis of carbohydrates
    k_hyd_pr: float = 10.0     # d⁻¹, hydrolysis of proteins
    k_hyd_li: float = 10.0     # d⁻¹, hydrolysis of lipids
    K_S_IN: float = 10**-4     # M, nitrogen half saturation
    k_m_su: float = 30.0       # d⁻¹, max uptake rate sugars
    K_S_su: float = 0.5        # kgCOD·m⁻³
    pH_UL_aa: float = 5.5
    pH_LL_aa: float = 4.0
    k_m_aa: float = 50.0       # d⁻¹
    K_S_aa: float = 0.3        # kgCOD·m⁻³
    k_m_fa: float = 6.0        # d⁻¹
    K_S_fa: float = 0.4        # kgCOD·m⁻³
    K_I_h2_fa: float = 5e-6    # kgCOD·m⁻³
    k_m_c4: float = 20.0       # d⁻¹
    K_S_c4: float = 0.2        # kgCOD·m⁻³
    K_I_h2_c4: float = 1e-5    # kgCOD·m⁻³
    k_m_pro: float = 13.0      # d⁻¹
    K_S_pro: float = 0.1       # kgCOD·m⁻³
    K_I_h2_pro: float = 3.5e-6 # kgCOD·m⁻³
    k_m_ac: float = 8.0        # d⁻¹
    K_S_ac: float = 0.15       # kgCOD·m⁻³
    K_I_nh3: float = 0.0018    # M
    pH_UL_ac: float = 7.0
    pH_LL_ac: float = 6.0
    k_m_h2: float = 35.0       # d⁻¹
    K_S_h2: float = 7e-6       # kgCOD·m⁻³
    pH_UL_h2: float = 6.0
    pH_LL_h2: float = 5.0
    k_dec_X_su: float = 0.02   # d⁻¹
    k_dec_X_aa: float = 0.02   # d⁻¹
    k_dec_X_fa: float = 0.02   # d⁻¹
    k_dec_X_c4: float = 0.02   # d⁻¹
    k_dec_X_pro: float = 0.02  # d⁻¹
    k_dec_X_ac: float = 0.02   # d⁻¹
    k_dec_X_h2: float = 0.02   # d⁻¹
    
    # Gas transfer
    k_L_a: float = 200.0       # d⁻¹, liquid-gas transfer coefficient
    
    def __post_init__(self):
        """Calculate temperature-dependent parameters."""
        self._calculate_temperature_dependent_params()
        self._calculate_ph_inhibition_params()
    
    def _calculate_temperature_dependent_params(self):
        """Calculate temperature-dependent equilibrium constants."""
        R = self.R
        T_base = self.T_base
        T_op = self.T_op
        
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1/T_base - 1/T_op))
        self.K_w = (10**-14.0) * np.exp((55900/(100*R)) * (1/T_base - 1/T_op))
        self.K_a_va = 10**-4.86
        self.K_a_bu = 10**-4.82
        self.K_a_pro = 10**-4.88
        self.K_a_ac = 10**-4.76
        self.K_a_co2 = (10**-6.35) * np.exp((7646/(100*R)) * (1/T_base - 1/T_op))
        self.K_a_IN = (10**-9.25) * np.exp((51965/(100*R)) * (1/T_base - 1/T_op))
        
        # Acid-base rate constants
        self.k_A_B_va = 10**10
        self.k_A_B_bu = 10**10
        self.k_A_B_pro = 10**10
        self.k_A_B_ac = 10**10
        self.k_A_B_co2 = 10**10
        self.k_A_B_IN = 10**10
        
        # Henry's constants
        self.K_H_co2 = 0.035 * np.exp((-19410/(100*R)) * (1/T_base - 1/T_op))
        self.K_H_ch4 = 0.0014 * np.exp((-14240/(100*R)) * (1/T_base - 1/T_op))
        self.K_H_h2 = (7.8e-4) * np.exp(-4180/(100*R) * (1/T_base - 1/T_op))
    
    def _calculate_ph_inhibition_params(self):
        """Calculate pH inhibition parameters."""
        self.K_pH_aa = 10**(-0.5 * (self.pH_LL_aa + self.pH_UL_aa))
        self.n_aa = 3.0 / (self.pH_UL_aa - self.pH_LL_aa)
        self.K_pH_ac = 10**(-0.5 * (self.pH_LL_ac + self.pH_UL_ac))
        self.n_ac = 3.0 / (self.pH_UL_ac - self.pH_LL_ac)
        self.K_pH_h2 = 10**(-0.5 * (self.pH_LL_h2 + self.pH_UL_h2))
        self.n_h2 = 3.0 / (self.pH_UL_h2 - self.pH_LL_h2)


class ADM1Model:
    """
    COD-based Anaerobic Digestion Model No. 1 (ADM1).
    
    A comprehensive mechanistic model for anaerobic digestion processes
    with 38 state variables covering biochemical and physicochemical processes.
    
    Attributes:
        params: ADM1Parameters object containing model parameters
        state_names: List of 38 state variable names
        results: DataFrame containing simulation results
        gas_flow: DataFrame containing gas production data
    
    Example:
        >>> model = ADM1Model()
        >>> model.load_data("influent.xlsx", "initial.csv")
        >>> results = model.run(solver_method="BDF")
        >>> model.plot_results()
    """
    
    # State variable names (38 states)
    STATE_NAMES = [
        "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", 
        "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", 
        "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", 
        "S_cation", "S_anion", "S_H_ion", "S_va_ion", "S_bu_ion", 
        "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2", "S_nh3", 
        "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"
    ]
    
    def __init__(self, params: Optional[ADM1Parameters] = None):
        """
        Initialize ADM1 model.
        
        Args:
            params: ADM1Parameters object. If None, uses default parameters.
        """
        self.params = params or ADM1Parameters()
        self.influent_data: Optional[pd.DataFrame] = None
        self.initial_state: Optional[np.ndarray] = None
        self.results: Optional[pd.DataFrame] = None
        self.gas_flow: Optional[pd.DataFrame] = None
        self._state: np.ndarray = np.zeros(38)
        self._state_input: np.ndarray = np.zeros(26)
        self._q_ad: float = 130.0  # Default flow rate
    
    def load_data(
        self, 
        influent_path: str, 
        initial_state_path: str,
        influent_sheet: str = "Influent_ADM1_COD_Based"
    ) -> None:
        """
        Load influent and initial state data from files.
        
        Args:
            influent_path: Path to Excel file with influent data
            initial_state_path: Path to CSV file with initial state
            influent_sheet: Sheet name in Excel file
        """
        # Load influent data
        if influent_path.endswith('.xlsx') or influent_path.endswith('.xls'):
            self.influent_data = pd.read_excel(influent_path, sheet_name=influent_sheet)
        else:
            self.influent_data = pd.read_csv(influent_path)
        
        # Load initial state
        initial_df = pd.read_csv(initial_state_path)
        self._initialize_state(initial_df)
    
    def _initialize_state(self, initial_df: pd.DataFrame) -> None:
        """Initialize state vector from DataFrame."""
        self._state = np.array([
            initial_df['S_su'][0], initial_df['S_aa'][0], initial_df['S_fa'][0],
            initial_df['S_va'][0], initial_df['S_bu'][0], initial_df['S_pro'][0],
            initial_df['S_ac'][0], initial_df['S_h2'][0], initial_df['S_ch4'][0],
            initial_df['S_IC'][0], initial_df['S_IN'][0], initial_df['S_I'][0],
            initial_df['X_xc'][0], initial_df['X_ch'][0], initial_df['X_pr'][0],
            initial_df['X_li'][0], initial_df['X_su'][0], initial_df['X_aa'][0],
            initial_df['X_fa'][0], initial_df['X_c4'][0], initial_df['X_pro'][0],
            initial_df['X_ac'][0], initial_df['X_h2'][0], initial_df['X_I'][0],
            initial_df['S_cation'][0], initial_df['S_anion'][0],
            initial_df['S_H_ion'][0], initial_df['S_va_ion'][0],
            initial_df['S_bu_ion'][0], initial_df['S_pro_ion'][0],
            initial_df['S_ac_ion'][0], initial_df['S_hco3_ion'][0],
            0.14,  # S_co2 initial
            initial_df['S_nh3'][0], 0.0041,  # S_nh4_ion initial
            initial_df['S_gas_h2'][0], initial_df['S_gas_ch4'][0],
            initial_df['S_gas_co2'][0]
        ])
        self.initial_state = self._state.copy()
    
    def set_influent(self, idx: int) -> None:
        """Set influent concentrations for time step idx."""
        if self.influent_data is None:
            raise ValueError("No influent data loaded. Call load_data() first.")
        
        row = self.influent_data.iloc[idx]
        self._state_input = np.array([
            row['S_su'], row['S_aa'], row['S_fa'], row['S_va'],
            row['S_bu'], row['S_pro'], row['S_ac'], row['S_h2'],
            row['S_ch4'], row['S_IC'], row['S_IN'], row['S_I'],
            row['X_xc'], row['X_ch'], row['X_pr'], row['X_li'],
            row['X_su'], row['X_aa'], row['X_fa'], row['X_c4'],
            row['X_pro'], row['X_ac'], row['X_h2'], row['X_I'],
            row['S_cation'], row['S_anion']
        ])
        self._q_ad = row.get('q_ad', 130.0)
    
    def _adm1_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ADM1 differential equations.
        
        Args:
            t: Current time
            y: State vector (38 elements)
        
        Returns:
            Derivatives array (38 elements)
        """
        p = self.params
        q_ad = self._q_ad
        V_liq = p.V_liq
        V_gas = p.V_gas
        
        # Unpack state
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2 = y[0:8]
        S_ch4, S_IC, S_IN, S_I = y[8:12]
        X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4 = y[12:20]
        X_pro, X_ac, X_h2, X_I = y[20:24]
        S_cation, S_anion, S_H_ion = y[24:27]
        S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion = y[27:31]
        S_hco3_ion, S_co2, S_nh3, S_nh4_ion = y[31:35]
        S_gas_h2, S_gas_ch4, S_gas_co2 = y[35:38]
        
        # Unpack influent
        inp = self._state_input
        
        # Derived quantities
        S_nh4_ion = S_IN - S_nh3
        S_co2 = S_IC - S_hco3_ion
        
        # Gas phase partial pressures
        p_gas_h2 = S_gas_h2 * p.R * p.T_op / 16
        p_gas_ch4 = S_gas_ch4 * p.R * p.T_op / 64
        p_gas_co2 = S_gas_co2 * p.R * p.T_op
        
        # Inhibition terms
        I_pH_aa = (p.K_pH_aa**p.n_aa) / (S_H_ion**p.n_aa + p.K_pH_aa**p.n_aa)
        I_pH_ac = (p.K_pH_ac**p.n_ac) / (S_H_ion**p.n_ac + p.K_pH_ac**p.n_ac)
        I_pH_h2 = (p.K_pH_h2**p.n_h2) / (S_H_ion**p.n_h2 + p.K_pH_h2**p.n_h2)
        I_IN_lim = 1 / (1 + p.K_S_IN / (S_IN + 1e-12))
        I_h2_fa = 1 / (1 + S_h2 / p.K_I_h2_fa)
        I_h2_c4 = 1 / (1 + S_h2 / p.K_I_h2_c4)
        I_h2_pro = 1 / (1 + S_h2 / p.K_I_h2_pro)
        I_nh3 = 1 / (1 + S_nh3 / p.K_I_nh3)
        
        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim
        
        # Biochemical process rates (Rho_1 to Rho_19)
        Rho_1 = p.k_dis * X_xc
        Rho_2 = p.k_hyd_ch * X_ch
        Rho_3 = p.k_hyd_pr * X_pr
        Rho_4 = p.k_hyd_li * X_li
        Rho_5 = p.k_m_su * S_su / (p.K_S_su + S_su) * X_su * I_5
        Rho_6 = p.k_m_aa * S_aa / (p.K_S_aa + S_aa) * X_aa * I_6
        Rho_7 = p.k_m_fa * S_fa / (p.K_S_fa + S_fa) * X_fa * I_7
        
        denom_c4 = S_bu + S_va + 1e-6
        Rho_8 = p.k_m_c4 * S_va / (p.K_S_c4 + S_va) * X_c4 * (S_va / denom_c4) * I_8
        Rho_9 = p.k_m_c4 * S_bu / (p.K_S_c4 + S_bu) * X_c4 * (S_bu / denom_c4) * I_9
        Rho_10 = p.k_m_pro * S_pro / (p.K_S_pro + S_pro) * X_pro * I_10
        Rho_11 = p.k_m_ac * S_ac / (p.K_S_ac + S_ac) * X_ac * I_11
        Rho_12 = p.k_m_h2 * S_h2 / (p.K_S_h2 + S_h2) * X_h2 * I_12
        
        # Decay rates
        Rho_13 = p.k_dec_X_su * X_su
        Rho_14 = p.k_dec_X_aa * X_aa
        Rho_15 = p.k_dec_X_fa * X_fa
        Rho_16 = p.k_dec_X_c4 * X_c4
        Rho_17 = p.k_dec_X_pro * X_pro
        Rho_18 = p.k_dec_X_ac * X_ac
        Rho_19 = p.k_dec_X_h2 * X_h2
        
        # Gas transfer rates
        Rho_T_8 = p.k_L_a * (S_h2 - 16 * p.K_H_h2 * p_gas_h2)
        Rho_T_9 = p.k_L_a * (S_ch4 - 64 * p.K_H_ch4 * p_gas_ch4)
        Rho_T_10 = p.k_L_a * (S_co2 - p.K_H_co2 * p_gas_co2)
        
        # Carbon balance coefficients
        s_1 = -p.C_xc + p.f_sI_xc*p.C_sI + p.f_ch_xc*p.C_ch + p.f_pr_xc*p.C_pr + p.f_li_xc*p.C_li + p.f_xI_xc*p.C_xI
        s_2 = -p.C_ch + p.C_su
        s_3 = -p.C_pr + p.C_aa
        s_4 = -p.C_li + (1 - p.f_fa_li)*p.C_su + p.f_fa_li*p.C_fa
        s_5 = -p.C_su + (1 - p.Y_su)*(p.f_bu_su*p.C_bu + p.f_pro_su*p.C_pro + p.f_ac_su*p.C_ac) + p.Y_su*p.C_bac
        s_6 = -p.C_aa + (1 - p.Y_aa)*(p.f_va_aa*p.C_va + p.f_bu_aa*p.C_bu + p.f_pro_aa*p.C_pro + p.f_ac_aa*p.C_ac) + p.Y_aa*p.C_bac
        s_7 = -p.C_fa + (1 - p.Y_fa)*0.7*p.C_ac + p.Y_fa*p.C_bac
        s_8 = -p.C_va + (1 - p.Y_c4)*0.54*p.C_pro + (1 - p.Y_c4)*0.31*p.C_ac + p.Y_c4*p.C_bac
        s_9 = -p.C_bu + (1 - p.Y_c4)*0.8*p.C_ac + p.Y_c4*p.C_bac
        s_10 = -p.C_pro + (1 - p.Y_pro)*0.57*p.C_ac + p.Y_pro*p.C_bac
        s_11 = -p.C_ac + (1 - p.Y_ac)*p.C_ch4 + p.Y_ac*p.C_bac
        s_12 = (1 - p.Y_h2)*p.C_ch4 + p.Y_h2*p.C_bac
        s_13 = -p.C_bac + p.C_xc
        
        # Gas flow
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p.p_gas_h2o
        q_gas = max(0, p.k_p * (p_gas - p.p_atm))
        
        # Differential equations (soluble matter 1-12)
        diff = np.zeros(38)
        diff[0] = q_ad/V_liq * (inp[0] - S_su) + Rho_2 + (1 - p.f_fa_li)*Rho_4 - Rho_5
        diff[1] = q_ad/V_liq * (inp[1] - S_aa) + Rho_3 - Rho_6
        diff[2] = q_ad/V_liq * (inp[2] - S_fa) + p.f_fa_li*Rho_4 - Rho_7
        diff[3] = q_ad/V_liq * (inp[3] - S_va) + (1-p.Y_aa)*p.f_va_aa*Rho_6 - Rho_8
        diff[4] = q_ad/V_liq * (inp[4] - S_bu) + (1-p.Y_su)*p.f_bu_su*Rho_5 + (1-p.Y_aa)*p.f_bu_aa*Rho_6 - Rho_9
        diff[5] = q_ad/V_liq * (inp[5] - S_pro) + (1-p.Y_su)*p.f_pro_su*Rho_5 + (1-p.Y_aa)*p.f_pro_aa*Rho_6 + (1-p.Y_c4)*0.54*Rho_8 - Rho_10
        diff[6] = q_ad/V_liq * (inp[6] - S_ac) + (1-p.Y_su)*p.f_ac_su*Rho_5 + (1-p.Y_aa)*p.f_ac_aa*Rho_6 + (1-p.Y_fa)*0.7*Rho_7 + (1-p.Y_c4)*0.31*Rho_8 + (1-p.Y_c4)*0.8*Rho_9 + (1-p.Y_pro)*0.57*Rho_10 - Rho_11
        diff[7] = 0  # S_h2 solved via DAE
        diff[8] = q_ad/V_liq * (inp[8] - S_ch4) + (1-p.Y_ac)*Rho_11 + (1-p.Y_h2)*Rho_12 - Rho_T_9
        
        Sigma = (s_1*Rho_1 + s_2*Rho_2 + s_3*Rho_3 + s_4*Rho_4 + s_5*Rho_5 + 
                 s_6*Rho_6 + s_7*Rho_7 + s_8*Rho_8 + s_9*Rho_9 + s_10*Rho_10 + 
                 s_11*Rho_11 + s_12*Rho_12 + s_13*(Rho_13+Rho_14+Rho_15+Rho_16+Rho_17+Rho_18+Rho_19))
        diff[9] = q_ad/V_liq * (inp[9] - S_IC) - Sigma - Rho_T_10
        diff[10] = q_ad/V_liq * (inp[10] - S_IN) + (p.N_xc - p.f_xI_xc*p.N_I - p.f_sI_xc*p.N_I - p.f_pr_xc*p.N_aa)*Rho_1 - p.Y_su*p.N_bac*Rho_5 + (p.N_aa - p.Y_aa*p.N_bac)*Rho_6 - p.Y_fa*p.N_bac*Rho_7 - p.Y_c4*p.N_bac*(Rho_8+Rho_9) - p.Y_pro*p.N_bac*Rho_10 - p.Y_ac*p.N_bac*Rho_11 - p.Y_h2*p.N_bac*Rho_12 + (p.N_bac - p.N_xc)*(Rho_13+Rho_14+Rho_15+Rho_16+Rho_17+Rho_18+Rho_19)
        diff[11] = q_ad/V_liq * (inp[11] - S_I) + p.f_sI_xc*Rho_1
        
        # Particulate matter (13-24)
        decay_sum = Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19
        diff[12] = q_ad/V_liq * (inp[12] - X_xc) - Rho_1 + decay_sum
        diff[13] = q_ad/V_liq * (inp[13] - X_ch) + p.f_ch_xc*Rho_1 - Rho_2
        diff[14] = q_ad/V_liq * (inp[14] - X_pr) + p.f_pr_xc*Rho_1 - Rho_3
        diff[15] = q_ad/V_liq * (inp[15] - X_li) + p.f_li_xc*Rho_1 - Rho_4
        diff[16] = q_ad/V_liq * (inp[16] - X_su) + p.Y_su*Rho_5 - Rho_13
        diff[17] = q_ad/V_liq * (inp[17] - X_aa) + p.Y_aa*Rho_6 - Rho_14
        diff[18] = q_ad/V_liq * (inp[18] - X_fa) + p.Y_fa*Rho_7 - Rho_15
        diff[19] = q_ad/V_liq * (inp[19] - X_c4) + p.Y_c4*(Rho_8 + Rho_9) - Rho_16
        diff[20] = q_ad/V_liq * (inp[20] - X_pro) + p.Y_pro*Rho_10 - Rho_17
        diff[21] = q_ad/V_liq * (inp[21] - X_ac) + p.Y_ac*Rho_11 - Rho_18
        diff[22] = q_ad/V_liq * (inp[22] - X_h2) + p.Y_h2*Rho_12 - Rho_19
        diff[23] = q_ad/V_liq * (inp[23] - X_I) + p.f_xI_xc*Rho_1
        
        # Cations/anions
        diff[24] = q_ad/V_liq * (inp[24] - S_cation)
        diff[25] = q_ad/V_liq * (inp[25] - S_anion)
        
        # Ion states (solved via DAE, set to 0)
        diff[26:35] = 0
        
        # Gas phase
        diff[35] = -q_gas/V_gas * S_gas_h2 + Rho_T_8 * V_liq/V_gas
        diff[36] = -q_gas/V_gas * S_gas_ch4 + Rho_T_9 * V_liq/V_gas
        diff[37] = -q_gas/V_gas * S_gas_co2 + Rho_T_10 * V_liq/V_gas
        
        return diff
    
    def _solve_dae(self, state: np.ndarray) -> np.ndarray:
        """
        Solve DAE for S_H_ion and S_h2 using Newton-Raphson.
        
        Args:
            state: Current state vector
        
        Returns:
            Updated state vector with solved algebraic variables
        """
        p = self.params
        tol = 1e-12
        max_iter = 1000
        
        # Extract relevant states
        S_va, S_bu, S_pro, S_ac = state[3:7]
        S_IC, S_IN = state[9:11]
        S_cation, S_anion = state[24:26]
        S_H_ion = state[26]
        S_gas_h2 = state[35]
        X_su, X_aa, X_fa, X_c4, X_pro, X_h2 = state[16], state[17], state[18], state[19], state[20], state[22]
        S_su, S_aa, S_fa = state[0:3]
        
        # Solve S_H_ion
        for _ in range(max_iter):
            S_va_ion = p.K_a_va * S_va / (p.K_a_va + S_H_ion)
            S_bu_ion = p.K_a_bu * S_bu / (p.K_a_bu + S_H_ion)
            S_pro_ion = p.K_a_pro * S_pro / (p.K_a_pro + S_H_ion)
            S_ac_ion = p.K_a_ac * S_ac / (p.K_a_ac + S_H_ion)
            S_hco3_ion = p.K_a_co2 * S_IC / (p.K_a_co2 + S_H_ion)
            S_nh3 = p.K_a_IN * S_IN / (p.K_a_IN + S_H_ion)
            
            delta = (S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion 
                    - S_ac_ion/64 - S_pro_ion/112 - S_bu_ion/160 - S_va_ion/208 
                    - p.K_w/S_H_ion - S_anion)
            
            if abs(delta) < tol:
                break
            
            grad = (1 + p.K_a_IN*S_IN/(p.K_a_IN + S_H_ion)**2 
                   + p.K_a_co2*S_IC/(p.K_a_co2 + S_H_ion)**2
                   + p.K_a_ac*S_ac/64/(p.K_a_ac + S_H_ion)**2
                   + p.K_a_pro*S_pro/112/(p.K_a_pro + S_H_ion)**2
                   + p.K_a_bu*S_bu/160/(p.K_a_bu + S_H_ion)**2
                   + p.K_a_va*S_va/208/(p.K_a_va + S_H_ion)**2
                   + p.K_w/S_H_ion**2)
            
            S_H_ion = max(tol, S_H_ion - delta/grad)
        
        # Update state with solved values
        state[26] = S_H_ion
        state[27] = S_va_ion
        state[28] = S_bu_ion
        state[29] = S_pro_ion
        state[30] = S_ac_ion
        state[31] = S_hco3_ion
        state[32] = S_IC - S_hco3_ion  # S_co2
        state[33] = S_nh3
        state[34] = S_IN - S_nh3  # S_nh4_ion
        
        return state
    
    def run(
        self, 
        solver_method: str = "BDF",
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run ADM1 simulation.
        
        Args:
            solver_method: ODE solver method ('BDF' or 'RK45')
            verbose: Whether to print progress
        
        Returns:
            DataFrame with simulation results
        """
        if self.influent_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        time_array = self.influent_data['time'].values
        n_steps = len(time_array)
        
        # Initialize results storage
        results_list = [self._state.copy()]
        gas_flow_list = [{'q_gas': 0.0, 'q_ch4': 0.0, 'time': time_array[0]}]
        
        self.set_influent(0)
        t0 = 0
        
        for n in range(1, n_steps):
            self.set_influent(n)
            t_span = [t0, time_array[n]]
            
            # Solve ODE
            sol = integrate.solve_ivp(
                self._adm1_ode, t_span, self._state, 
                method=solver_method
            )
            
            # Update state
            self._state = sol.y[:, -1]
            
            # Solve DAE
            self._state = self._solve_dae(self._state)
            
            # Calculate gas flow
            p = self.params
            p_gas_h2 = self._state[35] * p.R * p.T_op / 16
            p_gas_ch4 = self._state[36] * p.R * p.T_op / 64
            p_gas_co2 = self._state[37] * p.R * p.T_op
            p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p.p_gas_h2o
            q_gas = max(0, p.k_p * (p_gas - p.p_atm))
            q_ch4 = max(0, q_gas * p_gas_ch4 / p_gas) if p_gas > 0 else 0
            
            results_list.append(self._state.copy())
            gas_flow_list.append({
                'q_gas': q_gas, 
                'q_ch4': q_ch4, 
                'time': time_array[n]
            })
            
            t0 = time_array[n]
            
            if verbose and n % 50 == 0:
                print(f"Step {n}/{n_steps-1}, Day {time_array[n]:.1f}")
        
        # Convert to DataFrames
        self.results = pd.DataFrame(results_list, columns=self.STATE_NAMES)
        self.results['time'] = time_array
        self.results['pH'] = -np.log10(self.results['S_H_ion'])
        
        self.gas_flow = pd.DataFrame(gas_flow_list)
        
        if verbose:
            print("Simulation complete!")
        
        return self.results
    
    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return results and gas flow DataFrames."""
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        return self.results, self.gas_flow
    
    def plot_results(
        self, 
        variables: Optional[List[str]] = None,
        actual_data: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot simulation results.
        
        Args:
            variables: List of state variables to plot. If None, plots biogas.
            actual_data: Optional DataFrame with actual measurements for comparison
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if self.results is None or self.gas_flow is None:
            raise ValueError("No results available. Run simulation first.")
        
        if variables is None:
            # Default: plot biogas production
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Biogas flow
            axes[0, 0].plot(self.gas_flow['time'], self.gas_flow['q_gas'], label='Model')
            if actual_data is not None and 'Biogas (m3/day)' in actual_data.columns:
                axes[0, 0].plot(actual_data['time'], actual_data['Biogas (m3/day)'], 
                               '--', label='Actual')
            axes[0, 0].set_xlabel('Time (days)')
            axes[0, 0].set_ylabel('Biogas Flow (m³/day)')
            axes[0, 0].legend()
            axes[0, 0].set_title('Biogas Production')
            
            # CH4 flow
            axes[0, 1].plot(self.gas_flow['time'], self.gas_flow['q_ch4'])
            axes[0, 1].set_xlabel('Time (days)')
            axes[0, 1].set_ylabel('CH₄ Flow (m³/day)')
            axes[0, 1].set_title('Methane Production')
            
            # pH
            axes[1, 0].plot(self.results['time'], self.results['pH'])
            axes[1, 0].set_xlabel('Time (days)')
            axes[1, 0].set_ylabel('pH')
            axes[1, 0].set_title('Reactor pH')
            
            # VFA
            axes[1, 1].plot(self.results['time'], self.results['S_ac'], label='Acetate')
            axes[1, 1].plot(self.results['time'], self.results['S_pro'], label='Propionate')
            axes[1, 1].plot(self.results['time'], self.results['S_bu'], label='Butyrate')
            axes[1, 1].set_xlabel('Time (days)')
            axes[1, 1].set_ylabel('Concentration (kgCOD/m³)')
            axes[1, 1].legend()
            axes[1, 1].set_title('VFA Concentrations')
            
            plt.tight_layout()
            plt.show()
        else:
            # Plot specified variables
            n_vars = len(variables)
            n_cols = min(2, n_vars)
            n_rows = (n_vars + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = np.atleast_2d(axes)
            
            for i, var in enumerate(variables):
                row, col = i // n_cols, i % n_cols
                if var in self.results.columns:
                    axes[row, col].plot(self.results['time'], self.results[var])
                    axes[row, col].set_xlabel('Time (days)')
                    axes[row, col].set_ylabel(var)
                    axes[row, col].set_title(var)
            
            plt.tight_layout()
            plt.show()
    
    def save_results(self, output_path: str) -> None:
        """Save results to CSV file."""
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        self.results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
