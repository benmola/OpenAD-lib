"""
Feedstock module for OpenAD-lib.

Provides feedstock characterization and uncertainty quantification:
    - descriptors: Physical/chemical descriptors (TS, VS, C:N, BMP)
    - feedstock_library: Built-in database of common AD substrates
    - distributions: Probability distributions for uncertainty quantification
    - adm1_input_generator: Transformer for converting measurements to ADM1 inputs
"""

from openad_lib.feedstock.descriptors import FeedstockDescriptor, CoDigestionMixture
from openad_lib.feedstock.feedstock_library import FeedstockLibrary
from openad_lib.feedstock.distributions import (
    Distribution,
    BetaDistribution,
    LogNormalDistribution,
    GammaDistribution,
    assign_distribution
)

__all__ = [
    "FeedstockDescriptor",
    "CoDigestionMixture",
    "FeedstockLibrary",
    "Distribution",
    "BetaDistribution",
    "LogNormalDistribution",
    "GammaDistribution",
    "assign_distribution",
]
