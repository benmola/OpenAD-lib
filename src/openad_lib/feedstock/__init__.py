"""
Feedstock module for OpenAD-lib.

Provides feedstock characterization and uncertainty quantification:
    - descriptors: Physical/chemical descriptors (TS, VS, C:N, BMP)
    - feedstock_library: Built-in database of common AD substrates
    - adm1_input_generator: Transformer for converting measurements to ADM1 inputs
"""

from openad_lib.feedstock.descriptors import FeedstockDescriptor
from openad_lib.feedstock.feedstock_library import FeedstockLibrary

__all__ = [
    "FeedstockDescriptor",
    "FeedstockLibrary",
]
