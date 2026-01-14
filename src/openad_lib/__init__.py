"""
OpenAD-lib: Open-Source Framework for Uncertainty-Aware Anaerobic Digestion Digital Twins

A unified Python library for AD process modelling, control, and scheduling with 
explicit uncertainty quantification.

Modules:
    - feedstock: Feedstock characterization and uncertainty quantification
    - models.mechanistic: Mechanistic models (ADM1, AM2, etc.)
    - models.ml: Machine learning surrogate models (LSTM, GP, MTGP)
    - control: Control strategies (MPC)
    - optimisation: Parameter estimation and Bayesian optimization
    - utils: Utilities for data handling and visualization
"""

__version__ = "0.1.0"
__author__ = "Benaissa Dekhici"
__email__ = "b.dekhici@surrey.ac.uk"

# Lazy imports for optional submodules
def __getattr__(name):
    if name == "mechanistic":
        from openad_lib.models import mechanistic
        return mechanistic
    elif name == "ml":
        from openad_lib.models import ml
        return ml
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
]
