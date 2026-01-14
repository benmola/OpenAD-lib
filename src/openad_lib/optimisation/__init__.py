"""
Optimisation module for OpenAD-lib.

Provides optimization and calibration tools:
    - bayesian_doe: Bayesian experimental design
    - parameter_estimation: Model calibration routines
    - hyperparameter_tuning: ML model hyperparameter optimization
"""

from openad_lib.optimisation.am2_calibration import AM2Calibrator

__all__ = ["AM2Calibrator"]
