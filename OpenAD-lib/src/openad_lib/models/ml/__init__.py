"""
Machine Learning surrogate models for anaerobic digestion.

Available models:
    - LSTMModel: Long Short-Term Memory networks for time series prediction
    - MultitaskGP: Multi-task Gaussian Process for multi-output prediction
    - GPModel: Standard Gaussian Process regression
"""

# Lazy imports to avoid requiring torch/gpytorch at package load time
def __getattr__(name):
    if name == "LSTMModel":
        from openad_lib.models.ml.lstm_model import LSTMModel
        return LSTMModel
    elif name == "MultitaskGP":
        from openad_lib.models.ml.mtgp import MultitaskGP
        return MultitaskGP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LSTMModel",
    "MultitaskGP",
]
