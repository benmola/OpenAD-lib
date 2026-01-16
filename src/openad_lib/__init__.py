"""
OpenAD-lib: Open-Source Framework for Uncertainty-Aware Anaerobic Digestion Digital Twins

A unified Python library for AD process modelling, control, and scheduling with 
explicit uncertainty quantification.

Quick Start
-----------
>>> import openad_lib as oad
>>> model = oad.ADM1Model()
>>> calibrator = oad.ADM1Calibrator(model, measured_data, influent_data)

Modules
-------
- models.mechanistic: Mechanistic models (ADM1, AM2, etc.)
- models.ml: Machine learning surrogate models (LSTM, GP, MTGP)
- control: Control strategies (MPC)
- optimisation: Parameter estimation and Bayesian optimization
- preprocessing: Data preprocessing utilities (ACoD)
- data: Data loading, validation, and dataset classes
- utils: Utilities for metrics, plotting, and validation
- config: Configuration management
"""

__version__ = "0.2.0"
__author__ = "Benaissa Dekhici"
__email__ = "b.dekhici@surrey.ac.uk"

# Import configuration first
from openad_lib.config import config, Config

# Lazy imports for main classes to avoid heavy dependencies at import time
def __getattr__(name):
    """Lazy loading of main classes."""
    
    # Mechanistic models
    if name == "ADM1Model":
        from openad_lib.models.mechanistic import ADM1Model
        return ADM1Model
    elif name == "AM2Model":
        from openad_lib.models.mechanistic import AM2Model
        return AM2Model
    
    # ML models
    elif name == "LSTMModel":
        from openad_lib.models.ml import LSTMModel
        return LSTMModel
    elif name == "MultitaskGP":
        from openad_lib.models.ml import MultitaskGP
        return MultitaskGP
    
    # Calibrators
    elif name == "ADM1Calibrator":
        from openad_lib.optimisation import ADM1Calibrator
        return ADM1Calibrator
    elif name == "AM2Calibrator":
        from openad_lib.optimisation import AM2Calibrator
        return AM2Calibrator
    
    # Preprocessing
    elif name == "acod":
        from openad_lib.preprocessing import acod
        return acod
    
    # Data utilities
    elif name == "load_sample_data":
        from openad_lib.data.loaders import load_sample_data
        return load_sample_data
    elif name == "BiogasDataset":
        from openad_lib.data.datasets import BiogasDataset
        return BiogasDataset
    elif name == "FeedstockDataset":
        from openad_lib.data.datasets import FeedstockDataset
        return FeedstockDataset
    
    # Submodules (for backward compatibility and direct access)
    elif name == "mechanistic":
        from openad_lib.models import mechanistic
        return mechanistic
    elif name == "ml":
        from openad_lib.models import ml
        return ml
    elif name == "data":
        from openad_lib import data
        return data
    elif name == "utils":
        from openad_lib import utils
        return utils
    elif name == "preprocessing":
        from openad_lib import preprocessing
        return preprocessing
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version info
    "__version__",
    
    # Configuration
    "config",
    "Config",
    
    # Models
    "ADM1Model",
    "AM2Model",
    "LSTMModel",
    "MultitaskGP",
    
    # Calibration
    "ADM1Calibrator",
    "AM2Calibrator",
    
    # Preprocessing
    "acod",
    
    # Data utilities
    "load_sample_data",
    "BiogasDataset",
    "FeedstockDataset",
    
    # Submodules
    "mechanistic",
    "ml",
    "data",
    "utils",
    "preprocessing",
]

