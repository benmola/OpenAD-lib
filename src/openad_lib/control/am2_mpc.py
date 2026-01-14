"""
AM2 MPC Controller.

This module implements a Model Predictive Controller for the AM2 anaerobic
digestion model using the do-mpc library. It allows for setpoint tracking
or maximizing biogas production by manipulating the dilution rate.
"""

import numpy as np
import do_mpc
import casadi
from typing import Dict, List, Optional, Tuple, Union
from ..models.mechanistic import AM2Parameters

class AM2MPC:
    """
    Model Predictive Controller for the AM2 Model.
    
    Attributes:
        params (AM2Parameters): Model parameters.
        model (do_mpc.model.Model): do-mpc model instance.
        mpc (do_mpc.controller.MPC): do-mpc controller instance.
        simulator (do_mpc.simulator.Simulator): do-mpc simulator instance.
        estimator (do_mpc.estimator.Estimator): do-mpc estimator instance.
    """
    
    def __init__(self, params: Optional[AM2Parameters] = None):
        """
        Initialize the AM2 MPC controller.
        
        Args:
            params: AM2Parameters object. If None, uses default parameters.
        """
        self.params = params or AM2Parameters()
        self.model: Optional[do_mpc.model.Model] = None
        self.mpc: Optional[do_mpc.controller.MPC] = None
        self.simulator: Optional[do_mpc.simulator.Simulator] = None
        self.estimator: Optional[do_mpc.estimator.StateFeedback] = None
        
        # Setup model immediately
        self.setup_model()
        
    def setup_model(self) -> None:
        """
        Define the AM2 model equations using CasADi variables for do-mpc.
        """
        # 1. Initialize model
        model_type = 'continuous' # continuous or discrete
        self.model = do_mpc.model.Model(model_type)
        
        # 2. Define states
        # S1: Organic substrate (COD) [g/L]
        # X1: Acidogenic bacteria [g/L]
        # S2: VFA concentration [g/L]
        # X2: Methanogenic bacteria [g/L]
        S1 = self.model.set_variable(var_type='_x', var_name='S1')
        X1 = self.model.set_variable(var_type='_x', var_name='X1')
        S2 = self.model.set_variable(var_type='_x', var_name='S2')
        X2 = self.model.set_variable(var_type='_x', var_name='X2')
        
        # 3. Define inputs (control variables)
        # D: Dilution rate [d^-1]
        D = self.model.set_variable(var_type='_u', var_name='D')
        
        # 4. Define parameters (time-varying but known/estimated)
        # S1in: Inlet substrate concentration [g/L]
        S1in = self.model.set_variable(var_type='_tvp', var_name='S1in')
        # pH: Reactor pH (can be controlled externally or assumed constant)
        pH = self.model.set_variable(var_type='_tvp', var_name='pH')
        
        # 5. Define algebraic outputs (Auxiliary variables)
        # Q: Biogas production rate [L/d]
        # We need this as an expression for the cost function
        
        # Get parameters
        p = self.params
        
        # --- Kinetics ---
        # Monod kinetics for acidogenesis
        mu1 = p.m1 * (S1 / (S1 + p.K1))
        
        # Haldane kinetics for methanogenesis with pH inhibition
        mu2_base = p.m2 * (S2 / ((S2**2) / p.Ki + S2 + p.K2))
        
        # pH inhibition term: I_pH = exp(-4 * ((pH - pHH)/(pHH - pHL))^2)
        # Avoiding division by zero if limits are weird, but assuming valid inputs
        pH_lim_diff = p.pHH - p.pHL
        # Note: CasADi uses 'exp'
        pH_factor = casadi.exp(-4 * ((pH - p.pHH) / pH_lim_diff)**2)
        
        mu2 = mu2_base * (1.0 - pH_factor) # Wait, formula in am2_model.py was base - base*factor?
        # Let's check am2_model.py again.
        # Line 315: pH_factor = np.exp(...)
        # Line 316: MU2 = MU2_base - (MU2_base * pH_factor)
        # ... Wait, normally inhibition factor I ranges 0 to 1, and you multiply rate by I.
        # If I=1 (optimal), rate is max. If I=0 (inhibited), rate is 0.
        # The code in am2_model.py says: MU2 = MU2_base * (1 - pH_factor) ? 
        # Actually it says: MU2 = MU2_base - (MU2_base * pH_factor) = MU2_base * (1 - pH_factor).
        # And pH_factor is a gaussian bell curve centered at pHH?
        # Let's re-verify the pH factor equation in AM2 model. 
        # Typically pH inhibition is: I = 1 for opt, I -> 0 for bad. 
        # The equation in am2_model.py: pH_factor = exp(-4 * ...) which is a bell curve (1 at peak, 0 at tails).
        # So (1 - pH_factor) would be 0 at peak (optimal pH) and 1 at tails? That creates a minimum at optimal pH!
        # That looks like a BUG in am2_model.py potentially? 
        # Or maybe pH_factor is defined as INHIBITION factor (0 = no inhibition).
        # The Gaussian exp(...) -> 1 when pH = pHH.
        # So at pH = pHH, (1 - 1) = 0. So growth is ZERO at pHH (Upper Limit)? 
        # That seems wrong if pHH is "Upper pH Limit for inhibition". Usually pH optimal is in the middle.
        # Let's assume for now I should replicate am2_model.py exactly, but maybe I should fix it?
        # If I fix it here, I deviate from the simulator. Best to replicate the simulator logic for MPC prediction to match.
        # But wait, if am2_model logic is inverted, the MPC will try to steer AWAY from pHH.
        # Let's stick to the code in am2_model.py for consistency.
        
        # Actually, looking at common models (like ADM1), pH inhibition is often a function I_pH where mu = mu_max * I_pH.
        # The code in am2_model.py line 316: MU2 = MU2_base - (MU2_base * pH_factor).
        # If pH_factor is close to 1 (pH near pHH), MU2 -> 0.
        # If pH_factor is close to 0 (pH far from pHH), MU2 -> MU2_base.
        # So this implies pHH is a TOXIC point? "Upper pH limit for inhibition".
        # If pH is typically 7, and pHH is 8.5 (default), then at 7, factor is exp(-4*((7-8.5)/(8.5-5.5))^2) = exp(-4*(1/2)).
        # It's an inhibition term. I will copy it exactly.
        
        mu2 = mu2_base * (1.0 - pH_factor)
        
        # --- ODEs ---
        # dS1/dt = D*(S1in - S1) - k1*mu1*X1
        dS1_dt = D * (S1in - S1) - p.k1 * mu1 * X1
        
        # dX1/dt = (mu1 - alpha*D)*X1
        dX1_dt = (mu1 - p.Alpha * D) * X1
        
        # dS2/dt = D*(S2in - S2) + k2*mu1*X1 - k3*mu2*X2
        dS2_dt = D * (p.S2in - S2) + p.k2 * mu1 * X1 - p.k3 * mu2 * X2
        
        # dX2/dt = (mu2 - alpha*D)*X2
        dX2_dt = (mu2 - p.Alpha * D) * X2
        
        self.model.set_rhs('S1', dS1_dt)
        self.model.set_rhs('X1', dX1_dt)
        self.model.set_rhs('S2', dS2_dt)
        self.model.set_rhs('X2', dX2_dt)
        
        # --- Outputs ---
        # Q = k6 * mu2 * X2 * c
        Q_biogas = p.k6 * mu2 * X2 * p.c
        
        # Auxiliary expression for Q so we can minimize/maximize it
        self.model.set_expression(expr_name='Q', expr=Q_biogas)
        
        self.model.setup()
        
    def setup_controller(
        self, 
        sampling_time: float = 1.0, 
        horizon: int = 20,
        objective_type: str = 'maximize_biogas', # 'maximize_biogas' or 'tracking'
        tracking_variable: str = 'Q', # Variable to track (e.g., 'Q', 'S2')
        setpoint: float = 0.0, # Target value for tracking_variable (renamed from Q_setpoint)
        D_max: float = 1.0
    ) -> None:
        """
        Configure the MPC controller.
        
        Args:
            sampling_time: Time step size [days]
            horizon: Prediction horizon steps
            objective_type: 'maximize_biogas' or 'tracking'
            tracking_variable: Name of variable to track ('Q', 'S1', 'S2', etc.)
            setpoint: Setpoint value (only used if 'tracking')
            D_max: Maximum dilution rate constraint
        """
        if self.model is None:
            self.setup_model()
            
        self.mpc = do_mpc.controller.MPC(self.model)
        
        # Optimizer settings
        setup_mpc = {
            'n_horizon': horizon,
            't_step': sampling_time,
            'n_robust': 0,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)
        
        # Objective function
        
        # mterm: Terminal cost
        # lterm: Stage cost
        
        if objective_type == 'maximize_biogas':
            # Minimize negative biogas production
            _Q = self.model.aux['Q']
            lterm = -_Q 
            mterm = -_Q
        elif objective_type == 'tracking':
            # Minimize quadratic error from setpoint
            # Identify variable
            if tracking_variable in self.model.aux.keys():
                var = self.model.aux[tracking_variable]
            elif tracking_variable in self.model.x.keys():
                var = self.model.x[tracking_variable]
            else:
                raise ValueError(f"Tracking variable '{tracking_variable}' not found in model.")
                
            lterm = (var - setpoint)**2
            mterm = (var - setpoint)**2
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
            
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        
        # Penalize control changes (smoothness)
        self.mpc.set_rterm(D=1.0) # Penalty on delta D
        
        # Constraints
        # 0 <= D <= D_max
        self.mpc.bounds['lower','_u', 'D'] = 0.0
        self.mpc.bounds['upper','_u', 'D'] = D_max
        
        # State constraints (physical limits)
        self.mpc.bounds['lower','_x', 'S1'] = 0.0
        self.mpc.bounds['lower','_x', 'X1'] = 0.0
        self.mpc.bounds['lower','_x', 'S2'] = 0.0
        self.mpc.bounds['lower','_x', 'X2'] = 0.0
        
        # Scale variables for better numerics (optional but recommended)
        self.mpc.scaling['_x', 'S1'] = 10.0
        self.mpc.scaling['_x', 'S2'] = 10.0
        self.mpc.scaling['_x', 'X1'] = 1.0
        self.mpc.scaling['_x', 'X2'] = 1.0
        self.mpc.scaling['_u', 'D'] = 0.1
        
        # Setup uncertain parameters (S1in and pH)
        # We start by assuming they are known/constant for the prediction horizon
        # Users can update tvp_template in the loop
        p_template = self.mpc.get_tvp_template()
        
        def tvp_fun(t_now):
            # Default to constant parameter values if not updated externally
            # This is a placeholder; in the loop we can inject actual forecasts
            # For now, let's just return the current 'nominal' value
            return p_template
            
        self.mpc.set_tvp_fun(tvp_fun)
        
        # Removed set_uncertainty_values as S1in and pH are now _tvp
        # and do-mpc handles them via set_tvp_fun without requiring uncertainty setup for n_robust=0
        
        self.mpc.setup()
        
    def setup_simulator(self, sampling_time: float = 1.0) -> None:
        """
        Configure the simulator.
        
        Args:
            sampling_time: Time step size [days]
        """
        if self.model is None:
            self.setup_model()
            
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step=sampling_time)
        
        # Parameters for simulator
        p_template = self.simulator.get_tvp_template()
        def tvp_fun(t_now):
            return p_template
        self.simulator.set_tvp_fun(tvp_fun)
        
        self.simulator.setup()
        
    def set_initial_state(self, x0: np.ndarray) -> None:
        """
        Set initial state for MPC, Simulator and Estimator.
        
        Args:
            x0: Initial state vector [S1, X1, S2, X2]
        """
        if self.mpc is None:
            raise ValueError("MPC not setup. Call setup_controller() first.")
            
        self.mpc.x0 = x0
        self.simulator.x0 = x0
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.estimator.x0 = x0
        
    def run_step(self, S1in_val: float, pH_val: float) -> Tuple[float, Dict]:
        """
        Run one step of the closed-loop control.
        
        Args:
            S1in_val: Current inlet substrate concentration
            pH_val: Current pH
            
        Returns:
            u_opt: Optimal control input (D)
            y_next: Next state (from simulator)
        """
        if self.estimator is None:
            # First call defaults
            self.set_initial_state(np.array([10.0, 0.5, 30.0, 0.5]))
            
        # Update parameters for Prediction Model (MPC)
        # MPC template is a PowerStructure with _tvp group and time index
        mpc_tvp = self.mpc.get_tvp_template()
        mpc_tvp['_tvp', :, 'S1in'] = S1in_val
        mpc_tvp['_tvp', :, 'pH'] = pH_val
        
        def mpc_tvp_fun(t_now):
            return mpc_tvp
        self.mpc.set_tvp_fun(mpc_tvp_fun)
        
        # Update parameters for Simulator (Plant)
        # Simulator template is a simple Struct with variable names
        sim_tvp = self.simulator.get_tvp_template()
        sim_tvp['S1in'] = S1in_val
        sim_tvp['pH'] = pH_val
        
        def sim_tvp_fun(t_now):
            return sim_tvp
        self.simulator.set_tvp_fun(sim_tvp_fun)
        
        # 1. Get Control Action
        x0 = self.estimator.x0
        u0 = self.mpc.make_step(x0)
        
        # 2. Simulate Plant (one step)
        y_next = self.simulator.make_step(u0)
        
        # 3. Update Estimator (Assume perfect state feedback for now)
        self.estimator.x0 = y_next
        
        return float(u0), y_next
