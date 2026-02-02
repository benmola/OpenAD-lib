"""
Feedstock library with built-in database of common AD substrates.

Provides a comprehensive database of feedstock properties for common
substrates used in anaerobic digestion, based on literature values
and laboratory measurements.
Feedstock library for managing collections of feedstock descriptors.
"""

from typing import Dict, Optional, List, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
from openad_lib.feedstock.descriptors import FeedstockDescriptor, CoDigestionMixture


class FeedstockLibrary:
    """
    Built-in database of common AD feedstocks.
    
    Contains characterization data for agricultural residues, energy crops,
    food waste, and industrial by-products commonly used in biogas production.
    
    Attributes:
        feedstocks: Dictionary of FeedstockDescriptor objects
    
    Example:
        >>> lib = FeedstockLibrary()
        >>> maize = lib.get("Maize")
        >>> print(f"Maize BMP: {maize.bmp} NL CH4/kg VS")
        
        >>> # Create co-digestion mixture
        >>> mix = lib.create_mixture(
        ...     ["Maize", "Chicken Litter"],
        ...     [0.7, 0.3]
        ... )
    """
    
    # Default feedstock database from AcoD-Method
    # Point estimates represent mean values from multi-source data
    # Uncertainty values represent standard deviations from literature/lab measurements
    _DEFAULT_DATA = {
        'Maize': {
            'ts': 312.8, 'vs': 947.2, 'cod_total': 421.7, 'cod_soluble': 52.7,
            'acetate': 1.84, 'propionate': 0.0, 'butyrate': 0.0, 'valerate': 0.01,
            'proteins': 31.1, 'lipids': 7.7, 'tan': 457, 'bmp': 293,
            # Uncertainty (std dev from multi-source data)
            'ts_std': 4.0, 'vs_std': 3.5, 'bmp_std': 3.5, 'cod_total_std': 6.5,
            'proteins_std': 1.5, 'lipids_std': 0.8
        },
        'Wholecrop': {
            'ts': 374.3, 'vs': 930.0, 'cod_total': 400.0, 'cod_soluble': 60.0,
            'acetate': 2.0, 'propionate': 0.5, 'butyrate': 0.2, 'valerate': 0.1,
            'proteins': 30.0, 'lipids': 5.0, 'tan': 300, 'bmp': 320,
            'ts_std': 5.2, 'vs_std': 4.1, 'bmp_std': 4.2, 'cod_total_std': 7.0,
            'proteins_std': 1.8, 'lipids_std': 0.6
        },
        'Chicken Litter': {
            'ts': 613.3, 'vs': 870.4, 'cod_total': 350.0, 'cod_soluble': 50.0,
            'acetate': 1.5, 'propionate': 0.3, 'butyrate': 0.1, 'valerate': 0.05,
            'proteins': 25.0, 'lipids': 3.0, 'tan': 200, 'bmp': 280,
            'ts_std': 8.5, 'vs_std': 5.8, 'bmp_std': 3.8, 'cod_total_std': 6.2,
            'proteins_std': 1.4, 'lipids_std': 0.4
        },
        'Lactose': {
            'ts': 141.9, 'vs': 798.1, 'cod_total': 300.0, 'cod_soluble': 40.0,
            'acetate': 1.0, 'propionate': 0.2, 'butyrate': 0.05, 'valerate': 0.03,
            'proteins': 20.0, 'lipids': 2.0, 'tan': 150, 'bmp': 250,
            'ts_std': 2.8, 'vs_std': 3.2, 'bmp_std': 3.0, 'cod_total_std': 5.5,
            'proteins_std': 1.2, 'lipids_std': 0.3
        },
        'Apple Pomace': {
            'ts': 155.8, 'vs': 982.7, 'cod_total': 320.0, 'cod_soluble': 45.0,
            'acetate': 1.2, 'propionate': 0.3, 'butyrate': 0.1, 'valerate': 0.05,
            'proteins': 22.0, 'lipids': 2.5, 'tan': 180, 'bmp': 270,
            'ts_std': 3.1, 'vs_std': 2.9, 'bmp_std': 3.3, 'cod_total_std': 5.8,
            'proteins_std': 1.3, 'lipids_std': 0.4
        },
        'Rice Bran': {
            'ts': 733.0, 'vs': 920.0, 'cod_total': 380.0, 'cod_soluble': 55.0,
            'acetate': 1.8, 'propionate': 0.4, 'butyrate': 0.15, 'valerate': 0.08,
            'proteins': 28.0, 'lipids': 4.0, 'tan': 250, 'bmp': 300,
            'ts_std': 9.2, 'vs_std': 4.5, 'bmp_std': 3.9, 'cod_total_std': 6.8,
            'proteins_std': 1.6, 'lipids_std': 0.5
        },
        'Crimped Wheat': {
            'ts': 750.0, 'vs': 300.0, 'cod_total': 410.0, 'cod_soluble': 65.0,
            'acetate': 2.2, 'propionate': 0.6, 'butyrate': 0.25, 'valerate': 0.12,
            'proteins': 32.0, 'lipids': 6.0, 'tan': 350, 'bmp': 310,
            'ts_std': 9.5, 'vs_std': 2.5, 'bmp_std': 4.1, 'cod_total_std': 7.2,
            'proteins_std': 1.9, 'lipids_std': 0.7
        },
        'Grass': {
            'ts': 295.0, 'vs': 965.0, 'cod_total': 370.0, 'cod_soluble': 50.0,
            'acetate': 1.7, 'propionate': 0.5, 'butyrate': 0.2, 'valerate': 0.1,
            'proteins': 27.0, 'lipids': 4.5, 'tan': 270, 'bmp': 290,
            'ts_std': 4.5, 'vs_std': 3.8, 'bmp_std': 3.6, 'cod_total_std': 6.5,
            'proteins_std': 1.5, 'lipids_std': 0.6
        },
        'Corn Gluten/Rapemeal': {
            'ts': 877.0, 'vs': 845.9, 'cod_total': 390.0, 'cod_soluble': 60.0,
            'acetate': 2.0, 'propionate': 0.6, 'butyrate': 0.25, 'valerate': 0.12,
            'proteins': 30.0, 'lipids': 5.5, 'tan': 320, 'bmp': 300,
            'ts_std': 10.5, 'vs_std': 5.2, 'bmp_std': 3.9, 'cod_total_std': 6.9,
            'proteins_std': 1.7, 'lipids_std': 0.7
        },
        'Potatoes': {
            'ts': 159.4, 'vs': 936.1, 'cod_total': 340.0, 'cod_soluble': 45.0,
            'acetate': 1.5, 'propionate': 0.4, 'butyrate': 0.15, 'valerate': 0.08,
            'proteins': 24.0, 'lipids': 3.5, 'tan': 230, 'bmp': 260,
            'ts_std': 3.2, 'vs_std': 3.4, 'bmp_std': 3.2, 'cod_total_std': 6.0,
            'proteins_std': 1.4, 'lipids_std': 0.5
        },
        'FYM': {  # Farmyard Manure
            'ts': 294.7, 'vs': 956.0, 'cod_total': 360.0, 'cod_soluble': 50.0,
            'acetate': 1.8, 'propionate': 0.5, 'butyrate': 0.2, 'valerate': 0.1,
            'proteins': 26.0, 'lipids': 4.0, 'tan': 260, 'bmp': 280,
            'ts_std': 4.3, 'vs_std': 3.9, 'bmp_std': 3.7, 'cod_total_std': 6.4,
            'proteins_std': 1.5, 'lipids_std': 0.5
        },
        'Fodder Beet': {
            'ts': 618.0, 'vs': 370.0, 'cod_total': 330.0, 'cod_soluble': 40.0,
            'acetate': 1.6, 'propionate': 0.4, 'butyrate': 0.18, 'valerate': 0.09,
            'proteins': 23.0, 'lipids': 3.8, 'tan': 240, 'bmp': 270,
            'ts_std': 8.0, 'vs_std': 3.0, 'bmp_std': 3.4, 'cod_total_std': 5.9,
            'proteins_std': 1.3, 'lipids_std': 0.5
        }
    }
    
    # COD conversion factors (based on Yac, Ypro, Ybu, Yva, Ypr, Yli)
    _CONVERSION_FACTORS = {
        'Yac': 1.06667,   # Acetate COD conversion
        'Ypro': 1.513514, # Propionate COD conversion
        'Ybu': 1.818182,  # Butyrate COD conversion
        'Yva': 2.039216,  # Valerate COD conversion
        'Ypr': 1.53,      # Protein COD conversion
        'Yli': 2.878      # Lipid COD conversion
    }
    
    def __init__(self, custom_data: Optional[Dict[str, Dict]] = None):
        """
        Initialize feedstock library.
        
        Args:
            custom_data: Optional custom feedstock data to add/override defaults
        """
        self._data = self._DEFAULT_DATA.copy()
        
        if custom_data:
            self._data.update(custom_data)
        
        # Build FeedstockDescriptor objects
        self.feedstocks: Dict[str, FeedstockDescriptor] = {}
        self._build_descriptors()
    
    def _build_descriptors(self) -> None:
        """Build FeedstockDescriptor objects from data."""
        for name, props in self._data.items():
            self.feedstocks[name] = FeedstockDescriptor(
                name=name,
                ts=props.get('ts', 0.0),
                vs=props.get('vs', 0.0),
                bmp=props.get('bmp', 0.0),
                cod_total=props.get('cod_total', 0.0),
                cod_soluble=props.get('cod_soluble', 0.0),
                proteins=props.get('proteins', 0.0),
                lipids=props.get('lipids', 0.0),
                tan=props.get('tan', 0.0),
                acetate=props.get('acetate', 0.0),
                propionate=props.get('propionate', 0.0),
                butyrate=props.get('butyrate', 0.0),
                valerate=props.get('valerate', 0.0)
            )
    
    def get(self, name: str) -> FeedstockDescriptor:
        """
        Get feedstock by name.
        
        Args:
            name: Feedstock name
        
        Returns:
            FeedstockDescriptor object
        
        Raises:
            KeyError: If feedstock not found
        """
        if name not in self.feedstocks:
            available = list(self.feedstocks.keys())
            raise KeyError(f"Feedstock '{name}' not found. Available: {available}")
        return self.feedstocks[name]
    
    def list_feedstocks(self) -> List[str]:
        """Get list of available feedstock names."""
        return list(self.feedstocks.keys())
    
    def add_feedstock(
        self,
        name: str,
        ts: float = 0.0,
        vs: float = 0.0,
        bmp: float = 0.0,
        **kwargs
    ) -> FeedstockDescriptor:
        """
        Add custom feedstock to library.
        
        Args:
            name: Feedstock identifier
            ts: Total Solids [kg/m³]
            vs: Volatile Solids [g/kg TS]
            bmp: BMP [NL CH4/kg VS]
            **kwargs: Additional feedstock properties
        
        Returns:
            Created FeedstockDescriptor
        """
        descriptor = FeedstockDescriptor(
            name=name,
            ts=ts,
            vs=vs,
            bmp=bmp,
            **kwargs
        )
        self.feedstocks[name] = descriptor
        return descriptor
    
    def create_mixture(
        self,
        feedstock_names: List[str],
        ratios: List[float],
        mixture_name: str = "Mixture"
    ) -> FeedstockDescriptor:
        """
        Create co-digestion mixture from named feedstocks.
        
        Args:
            feedstock_names: List of feedstock names to mix
            ratios: Mixing ratios (should sum to 1.0)
            mixture_name: Name for the resulting mixture
        
        Returns:
            FeedstockDescriptor representing the mixture
        """
        feedstocks = [self.get(name) for name in feedstock_names]
        mixture = CoDigestionMixture(feedstocks, ratios)
        return mixture.to_descriptor(mixture_name)
    
    def get_probabilistic(
        self,
        name: str,
        user_data: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
        n_samples: int = 16
    ) -> FeedstockDescriptor:
        """
        Get probabilistic feedstock with fitted distributions, optionally merging user data.
        
        Creates a FeedstockDescriptor with probability distributions. If user provides
        measurements for some parameters, those are used; missing parameters are filled
        from the library's built-in uncertainty data.
        
        Args:
            name: Feedstock name
            user_data: Optional dict of user measurements, e.g. {'ts': [310, 315, 308]}
                      Only measured parameters need to be provided; library fills gaps
            n_samples: Number of synthetic samples to generate for library parameters (default: 16)
        
        Returns:
            FeedstockDescriptor with fitted probability distributions
            
        Raises:
            KeyError: If feedstock not found
            ValueError: If uncertainty data not available for feedstock
            
        Example:
            >>> lib = FeedstockLibrary()
            
            # Option 1: Use library data entirely (no user measurements)
            >>> maize = lib.get_probabilistic("Maize")
            
            # Option 2: Provide sparse user data, library fills gaps
            >>> user_measurements = {'ts': [310, 315, 308], 'bmp': [290, 295]}
            >>> maize = lib.get_probabilistic("Maize", user_data=user_measurements)
            # Uses user's TS and BMP, fills VS/COD/proteins from library
            
            # Option 3: Generate ensemble
            >>> ensemble = maize.sample(n=1000)
        """
        from openad_lib.feedstock.distributions import assign_distribution
        import numpy as np
        
        if name not in self._data:
            available = list(self.feedstocks.keys())
            raise KeyError(f"Feedstock '{name}' not found. Available: {available}")
        
        props = self._data[name]
        
        # Check if uncertainty data exists
        uncertainty_params = ['ts_std', 'vs_std', 'bmp_std', 'cod_total_std', 'proteins_std']
        if not any(param in props for param in uncertainty_params):
            raise ValueError(
                f"No uncertainty data available for '{name}'. "
                "Use get() for deterministic feedstock or add_feedstock() with uncertainty."
            )
        
        # Prepare samples dictionary
        samples_dict = {}
        user_data = user_data or {}  # Handle None case
        
        np.random.seed(42)  # Reproducible for library data
        
        for param in ['ts', 'vs', 'bmp', 'cod_total', 'proteins', 'lipids']:
            # Priority 1: Use user's measurements if provided
            if param in user_data:
                samples_dict[param] = np.asarray(user_data[param])
                continue
            
            # Priority 2: Generate from library uncertainty data
            mean_key = param
            std_key = f"{param}_std"
            
            if mean_key in props and std_key in props:
                mean_val = props[mean_key]
                std_val = props[std_key]
                
                # Generate synthetic samples (normal distribution)
                samples = np.random.normal(mean_val, std_val, n_samples)
                # Ensure positivity
                samples = np.maximum(samples, 0.1)
                
                samples_dict[param] = samples
        
        # Create probabilistic feedstock using from_samples
        feedstock_prob = FeedstockDescriptor.from_samples(
            name=f"{name} (Probabilistic)",
            samples_dict=samples_dict
        )
        
        return feedstock_prob
    
    def generate_ensemble(
        self,
        name: str,
        n_realizations: int = 1000,
        random_state: Optional[int] = None
    ) -> List[FeedstockDescriptor]:
        """
        Generate ensemble of feedstock realizations from built-in uncertainty data.
        
        Convenience method that combines get_probabilistic() and sample().
        
        Args:
            name: Feedstock name
            n_realizations: Number of ensemble members to generate
            random_state: Random seed for reproducibility
            
        Returns:
            List of FeedstockDescriptor realizations
            
        Example:
            >>> lib = FeedstockLibrary()
            >>> ensemble = lib.generate_ensemble("Maize", n_realizations=1000)
            >>> ts_values = [f.ts for f in ensemble]
            >>> print(f"TS: {np.mean(ts_values):.1f} ± {np.std(ts_values):.1f}")
        """
        feedstock_prob = self.get_probabilistic(name)
        return feedstock_prob.sample(n=n_realizations, random_state=random_state)

    
    def to_dataframe(self) -> pd.DataFrame:
        """Export library to pandas DataFrame."""
        records = [f.to_dict() for f in self.feedstocks.values()]
        return pd.DataFrame(records)
    
    @classmethod
    def from_csv(cls, path: str) -> 'FeedstockLibrary':
        """
        Load feedstock library from CSV file.
        
        Args:
            path: Path to CSV file with feedstock data
        
        Returns:
            FeedstockLibrary instance
        """
        df = pd.read_csv(path)
        
        custom_data = {}
        for _, row in df.iterrows():
            name = row.get('name', row.get('Substrate', f"Feedstock_{len(custom_data)}"))
            custom_data[name] = row.to_dict()
        
        return cls(custom_data=custom_data)
    
    def get_conversion_factors(self) -> Dict[str, float]:
        """Get COD conversion factors."""
        return self._CONVERSION_FACTORS.copy()
    
    def calculate_adm1_inputs(
        self,
        feedstock: FeedstockDescriptor,
        flow_rate: float
    ) -> Dict[str, float]:
        """
        Calculate ADM1 input concentrations from feedstock.
        
        Uses the transformer methodology to convert practical measurements
        to ADM1 state variable inputs.
        
        Args:
            feedstock: FeedstockDescriptor object
            flow_rate: Flow rate [m³/d]
        
        Returns:
            Dictionary of ADM1 input concentrations
        """
        cf = self._CONVERSION_FACTORS
        
        # Calculate biodegradable fraction
        if feedstock.cod_total > 0 and feedstock.vs > 0:
            f_d = (feedstock.bmp / (350 * feedstock.cod_total)) * feedstock.vs / 1000
        else:
            f_d = 0.8  # Default assumption
        
        f_d = min(max(f_d, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Inert fractions
        s_I = feedstock.cod_soluble * (1 - f_d) if feedstock.cod_soluble > 0 else 0
        x_I = feedstock.cod_particulate * (1 - f_d) if feedstock.cod_particulate > 0 else 0
        
        # Particulate organics
        x_li = feedstock.lipids * cf['Yli'] * f_d
        x_pr = feedstock.proteins * cf['Ypr'] * f_d
        x_ch = (feedstock.cod_particulate * f_d - x_pr - x_li) if feedstock.cod_particulate > 0 else 0
        x_ch = max(x_ch, 0)
        
        # Soluble VFA
        s_ac = feedstock.acetate * cf['Yac']
        s_pro = feedstock.propionate * cf['Ypro']
        s_bu = feedstock.butyrate * cf['Ybu']
        s_va = feedstock.valerate * cf['Yva']
        
        vfa_total = feedstock.acetate + feedstock.propionate + feedstock.butyrate + feedstock.valerate
        
        # Soluble organics (sugars, amino acids, fatty acids)
        if feedstock.cod_soluble > 0 and feedstock.cod_particulate > 0:
            sol_organic = (feedstock.cod_soluble * f_d) - vfa_total
            if sol_organic > 0 and (x_ch + x_pr + x_li) > 0:
                total_part = x_ch + x_pr + x_li
                s_su = sol_organic * (x_ch / total_part)
                s_aa = sol_organic * (x_pr / total_part)
                s_fa = sol_organic * (x_li / total_part)
            else:
                s_su = s_aa = s_fa = 0
        else:
            s_su = s_aa = s_fa = 0
        
        # Inorganic nitrogen
        M_N = 14.007
        s_IN = feedstock.tan / (M_N * 1000) if feedstock.tan > 0 else 0
        
        return {
            'S_su': max(s_su, 0),
            'S_aa': max(s_aa, 0),
            'S_fa': max(s_fa, 0),
            'S_va': s_va,
            'S_bu': s_bu,
            'S_pro': s_pro,
            'S_ac': s_ac,
            'S_h2': 0.0,
            'S_ch4': 0.0,
            'S_IC': 0.0,  # Usually needs to be set separately
            'S_IN': s_IN,
            'S_I': s_I,
            'X_xc': 0.0,  # Assume already disintegrated
            'X_ch': max(x_ch, 0),
            'X_pr': max(x_pr, 0),
            'X_li': max(x_li, 0),
            'X_su': 0.0,
            'X_aa': 0.0,
            'X_fa': 0.0,
            'X_c4': 0.0,
            'X_pro': 0.0,
            'X_ac': 0.0,
            'X_h2': 0.0,
            'X_I': x_I,
            'f_d': f_d,
            'flow_rate': flow_rate
        }
