"""
Models module for OpenAD-lib.

Contains both mechanistic and machine learning models:
    - mechanistic: Physics-based models (ADM1, AM2, etc.)
    - ml: Data-driven surrogate models (LSTM, GP, MTGP)
"""

# Lazy imports to avoid requiring optional dependencies
def __getattr__(name):
    if name == "mechanistic":
        from openad_lib.models import mechanistic
        return mechanistic
    elif name == "ml":
        from openad_lib.models import ml
        return ml
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["mechanistic", "ml"]
