"""
Feedstock descriptors for anaerobic digestion characterization.

Provides dataclasses and utilities for representing feedstock properties
including physical, chemical, and biochemical parameters with support
for uncertainty representation through probability distributions.

Example:
    >>> from openad_lib.feedstock import FeedstockDescriptor
    >>> maize = FeedstockDescriptor(
    ...     name="Maize Silage",
    ...     ts=312.8,  # kg/m³
    ...     vs=947.2,  # g/kg TS
    ...     bmp=293    # NL CH4/kg VS
    ... )
    >>> print(maize.cod_total)  # Calculate COD
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
import warnings


@dataclass
class FeedstockDescriptor:
    """
    Feedstock characterization descriptor for AD modeling.
    
    Stores physical, chemical, and biochemical properties of feedstocks
    used in anaerobic digestion processes. Supports both point estimates
    and uncertainty representation.
    
    Attributes:
        name: Feedstock identifier
        ts: Total Solids content [kg/m³ or %]
        vs: Volatile Solids [g/kg TS or % of TS]
        bmp: Biochemical Methane Potential [NL CH4/kg VS]
        cn_ratio: Carbon to Nitrogen ratio
        cod_total: Total COD [kg COD/m³]
        cod_soluble: Soluble COD [kg COD/m³]
        carbohydrates: Carbohydrate fraction [kg/m³]
        proteins: Protein fraction [kg/m³]
        lipids: Lipid fraction [kg/m³]
        tan: Total Ammonia Nitrogen [g N/m³]
        vfa: Volatile Fatty Acids [kg/m³]
        acetate: Acetate concentration [kg/m³]
        propionate: Propionate concentration [kg/m³]
        butyrate: Butyrate concentration [kg/m³]
        valerate: Valerate concentration [kg/m³]
    
    Example:
        >>> silage = FeedstockDescriptor(
        ...     name="Maize Silage",
        ...     ts=312.8,
        ...     vs=947.2,
        ...     bmp=293,
        ...     proteins=31.1,
        ...     lipids=7.7
        ... )
    """
    
    name: str
    ts: float = 0.0           # Total Solids [kg/m³]
    vs: float = 0.0           # Volatile Solids [g/kg TS]
    bmp: float = 0.0          # BMP [NL CH4/kg VS]
    cn_ratio: float = 0.0     # C:N ratio
    
    # COD fractions
    cod_total: float = 0.0    # Total COD [kg COD/m³]
    cod_soluble: float = 0.0  # Soluble COD [kg COD/m³]
    cod_particulate: float = 0.0  # Particulate COD [kg COD/m³]
    
    # Biochemical fractions
    carbohydrates: float = 0.0  # [kg/m³]
    proteins: float = 0.0       # [kg/m³]
    lipids: float = 0.0         # [kg/m³]
    fibre: float = 0.0          # [kg/m³]
    
    # Nitrogen
    tan: float = 0.0            # Total Ammonia Nitrogen [g N/m³]
    
    # VFA components
    vfa_total: float = 0.0      # Total VFA [kg/m³]
    acetate: float = 0.0        # [kg/m³]
    propionate: float = 0.0     # [kg/m³]
    butyrate: float = 0.0       # [kg/m³]
    valerate: float = 0.0       # [kg/m³]
    
    # Uncertainty (optional, for distribution-based representation)
    uncertainty: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and compute derived quantities."""
        if self.cod_particulate == 0.0 and self.cod_total > 0 and self.cod_soluble > 0:
            self.cod_particulate = self.cod_total - self.cod_soluble
        
        if self.vfa_total == 0.0:
            self.vfa_total = self.acetate + self.propionate + self.butyrate + self.valerate
    
    @property
    def vs_fraction(self) -> float:
        """VS as fraction of TS (0-1)."""
        if self.ts > 0:
            return (self.vs / 1000)  # Convert g/kg to fraction
        return 0.0
    
    @property
    def organic_fraction(self) -> float:
        """Total organic fraction (carbs + proteins + lipids)."""
        return self.carbohydrates + self.proteins + self.lipids
    
    def estimate_cod_from_composition(self) -> float:
        """
        Estimate COD from biochemical composition.
        
        Uses theoretical COD conversion factors:
        - Carbohydrates: 1.07 g COD/g
        - Proteins: 1.42 g COD/g  
        - Lipids: 2.90 g COD/g
        
        Returns:
            Estimated total COD [kg COD/m³]
        """
        cod = (1.07 * self.carbohydrates + 
               1.42 * self.proteins + 
               2.90 * self.lipids)
        return cod
    
    def estimate_cn_ratio(self, c_content: float = 0.45, n_protein: float = 0.16) -> float:
        """
        Estimate C:N ratio from protein content.
        
        Args:
            c_content: Carbon content as fraction of VS (default 0.45)
            n_protein: Nitrogen content of protein (default 0.16)
        
        Returns:
            Estimated C:N ratio
        """
        if self.proteins > 0:
            vs_kg = self.ts * (self.vs / 1000)  # VS in kg/m³
            c_total = vs_kg * c_content
            n_total = self.proteins * n_protein + self.tan / 1000
            if n_total > 0:
                return c_total / n_total
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'ts': self.ts,
            'vs': self.vs,
            'bmp': self.bmp,
            'cn_ratio': self.cn_ratio,
            'cod_total': self.cod_total,
            'cod_soluble': self.cod_soluble,
            'cod_particulate': self.cod_particulate,
            'carbohydrates': self.carbohydrates,
            'proteins': self.proteins,
            'lipids': self.lipids,
            'tan': self.tan,
            'vfa_total': self.vfa_total,
            'acetate': self.acetate,
            'propionate': self.propionate,
            'butyrate': self.butyrate,
            'valerate': self.valerate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedstockDescriptor':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_samples(
        cls,
        name: str,
        samples_dict: Dict[str, Union[List[float], np.ndarray]]
    ) -> 'FeedstockDescriptor':
        """
        Create FeedstockDescriptor with uncertainty from multi-source data.
        
        Fits probability distributions to observed samples using MLE.
        Automatically assigns appropriate distribution types based on
        parameter names.
        
        Args:
            name: Feedstock identifier
            samples_dict: Dictionary mapping parameter names to sample arrays
                         e.g., {'ts': [310, 315, 308], 'vs': [945, 950, 948]}
        
        Returns:
            FeedstockDescriptor with fitted distributions in uncertainty field
            
        Example:
            >>> samples = {
            ...     'ts': [310, 315, 308, 312],
            ...     'vs': [945, 950, 948, 947],
            ...     'bmp': [290, 295, 293, 292],
            ...     'biodegradability': [0.75, 0.82, 0.78, 0.80]
            ... }
            >>> feedstock = FeedstockDescriptor.from_samples("Maize", samples)
            >>> print(feedstock.uncertainty['ts'])
        """
        from openad_lib.feedstock.distributions import assign_distribution
        
        # Calculate point estimates (means)
        point_estimates = {}
        distributions = {}
        
        for param, samples in samples_dict.items():
            samples_array = np.asarray(samples)
            
            # Point estimate (mean)
            point_estimates[param] = float(np.mean(samples_array))
            
            # Fit distribution
            try:
                distributions[param] = assign_distribution(param, samples_array)
            except Exception as e:
                warnings.warn(f"Could not fit distribution for '{param}': {e}")
        
        # Create descriptor with point estimates
        descriptor = cls(name=name, **point_estimates)
        descriptor.uncertainty = distributions
        
        return descriptor
    
    def sample(
        self,
        n: int = 1,
        random_state: Optional[int] = None,
        apply_constraints: bool = True
    ) -> Union['FeedstockDescriptor', List['FeedstockDescriptor']]:
        """
        Generate ensemble of feedstock realizations from distributions.
        
        Samples from fitted probability distributions with optional
        physical constraint enforcement.
        
        Args:
            n: Number of samples to generate
            random_state: Random seed for reproducibility
            apply_constraints: If True, enforce physical constraints
            
        Returns:
            Single FeedstockDescriptor if n=1, else list of descriptors
            
        Raises:
            ValueError: If no uncertainty distributions are defined
            
        Example:
            >>> # Assuming feedstock has fitted distributions
            >>> ensemble = feedstock.sample(n=100, random_state=42)
            >>> print(f"Generated {len(ensemble)} realizations")
        """
        if self.uncertainty is None or len(self.uncertainty) == 0:
            raise ValueError(
                f"No uncertainty distributions defined for '{self.name}'. "
                "Use from_samples() to fit distributions first."
            )
        
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = []
        for i in range(n):
            # Sample from each distribution
            sample_dict = {'name': f"{self.name}_sample_{i}"}
            
            for param, distribution in self.uncertainty.items():
                sample_dict[param] = float(distribution.sample(1)[0])
            
            # Apply physical constraints if requested
            if apply_constraints:
                sample_dict = self._apply_constraints(sample_dict)
            
            samples.append(FeedstockDescriptor(**sample_dict))
        
        return samples[0] if n == 1 else samples
    
    def _apply_constraints(self, sample_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply physical consistency constraints to sampled parameters.
        
        Constraints:
        - VS ≤ 1000 g/kg TS (100%)
        - COD_soluble ≤ COD_total
        - All concentrations ≥ 0
        - VFA components ≤ VFA_total
        
        Args:
            sample_dict: Dictionary of sampled parameter values
            
        Returns:
            Constrained parameter dictionary
        """
        # Ensure non-negativity
        for key in sample_dict:
            if isinstance(sample_dict[key], (int, float)) and key != 'name':
                sample_dict[key] = max(sample_dict[key], 0.0)
        
        # VS ≤ 100% of TS
        if 'vs' in sample_dict:
            sample_dict['vs'] = min(sample_dict['vs'], 1000.0)
        
        # Soluble COD ≤ Total COD
        if 'cod_soluble' in sample_dict and 'cod_total' in sample_dict:
            sample_dict['cod_soluble'] = min(
                sample_dict['cod_soluble'],
                sample_dict['cod_total']
            )
        
        # Particulate COD consistency
        if 'cod_particulate' in sample_dict and 'cod_total' in sample_dict and 'cod_soluble' in sample_dict:
            sample_dict['cod_particulate'] = max(
                sample_dict['cod_total'] - sample_dict['cod_soluble'],
                0.0
            )
        
        # VFA components ≤ Total VFA
        vfa_components = ['acetate', 'propionate', 'butyrate', 'valerate']
        if 'vfa_total' in sample_dict:
            total_vfa = sum(sample_dict.get(vfa, 0.0) for vfa in vfa_components)
            if total_vfa > sample_dict['vfa_total']:
                # Scale down proportionally
                scale = sample_dict['vfa_total'] / total_vfa
                for vfa in vfa_components:
                    if vfa in sample_dict:
                        sample_dict[vfa] *= scale
        
        return sample_dict
    
    def get_distribution(self, parameter: str):
        """
        Get probability distribution for a specific parameter.
        
        Args:
            parameter: Parameter name
            
        Returns:
            Distribution object
            
        Raises:
            KeyError: If parameter has no fitted distribution
        """
        if self.uncertainty is None:
            raise ValueError(f"No uncertainty distributions defined for '{self.name}'")
        
        if parameter not in self.uncertainty:
            raise KeyError(
                f"No distribution for parameter '{parameter}'. "
                f"Available: {list(self.uncertainty.keys())}"
            )
        
        return self.uncertainty[parameter]



@dataclass  
class CoDigestionMixture:
    """
    Representation of a co-digestion feedstock mixture.
    
    Combines multiple feedstocks with specified mixing ratios
    to calculate weighted average properties.
    
    Attributes:
        feedstocks: List of FeedstockDescriptor objects
        ratios: Mixing ratios (should sum to 1.0)
    
    Example:
        >>> maize = FeedstockDescriptor(name="Maize", ts=312.8, vs=947.2, bmp=293)
        >>> chicken = FeedstockDescriptor(name="Chicken Litter", ts=613.3, vs=870.4, bmp=280)
        >>> mix = CoDigestionMixture([maize, chicken], [0.7, 0.3])
        >>> print(mix.weighted_bmp)
    """
    
    feedstocks: list = field(default_factory=list)
    ratios: list = field(default_factory=list)
    
    def __post_init__(self):
        """Validate ratios."""
        if len(self.feedstocks) != len(self.ratios):
            raise ValueError("Number of feedstocks must match number of ratios")
        
        total = sum(self.ratios)
        if abs(total - 1.0) > 0.01:
            warnings.warn(f"Ratios sum to {total}, normalizing to 1.0")
            self.ratios = [r / total for r in self.ratios]
    
    def _weighted_property(self, prop: str) -> float:
        """Calculate weighted average of a property."""
        total = 0.0
        for feedstock, ratio in zip(self.feedstocks, self.ratios):
            value = getattr(feedstock, prop, 0.0)
            total += value * ratio
        return total
    
    @property
    def weighted_ts(self) -> float:
        """Weighted TS [kg/m³]."""
        return self._weighted_property('ts')
    
    @property
    def weighted_vs(self) -> float:
        """Weighted VS [g/kg TS]."""
        return self._weighted_property('vs')
    
    @property
    def weighted_bmp(self) -> float:
        """Weighted BMP [NL CH4/kg VS]."""
        return self._weighted_property('bmp')
    
    @property
    def weighted_cod_total(self) -> float:
        """Weighted total COD [kg COD/m³]."""
        return self._weighted_property('cod_total')
    
    @property
    def weighted_proteins(self) -> float:
        """Weighted proteins [kg/m³]."""
        return self._weighted_property('proteins')
    
    @property
    def weighted_lipids(self) -> float:
        """Weighted lipids [kg/m³]."""
        return self._weighted_property('lipids')
    
    @property
    def weighted_carbohydrates(self) -> float:
        """Weighted carbohydrates [kg/m³]."""
        return self._weighted_property('carbohydrates')
    
    def to_descriptor(self, name: str = "Mixture") -> FeedstockDescriptor:
        """Convert mixture to single FeedstockDescriptor with weighted values."""
        return FeedstockDescriptor(
            name=name,
            ts=self.weighted_ts,
            vs=self.weighted_vs,
            bmp=self.weighted_bmp,
            cod_total=self.weighted_cod_total,
            cod_soluble=self._weighted_property('cod_soluble'),
            proteins=self.weighted_proteins,
            lipids=self.weighted_lipids,
            carbohydrates=self.weighted_carbohydrates,
            tan=self._weighted_property('tan'),
            acetate=self._weighted_property('acetate'),
            propionate=self._weighted_property('propionate'),
            butyrate=self._weighted_property('butyrate'),
            valerate=self._weighted_property('valerate')
        )
