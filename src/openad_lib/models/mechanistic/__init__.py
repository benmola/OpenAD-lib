"""
Mechanistic models for anaerobic digestion simulation.

Available models:
    - ADM1Model: Complete Anaerobic Digestion Model No. 1 (COD-based, 38 states)
    - AM2Model: Simplified Two-Step Anaerobic Digestion Model (5 states)
"""

from openad_lib.models.mechanistic.adm1_model import ADM1Model
from openad_lib.models.mechanistic.am2_model import AM2Model, AM2Parameters

__all__ = [
    "ADM1Model",
    "AM2Model",
    "AM2Parameters",
]
