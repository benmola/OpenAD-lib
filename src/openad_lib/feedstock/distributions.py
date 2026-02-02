"""
Probability distributions for feedstock uncertainty quantification.

Provides distribution classes for representing parameter uncertainty in
anaerobic digestion feedstocks, with automatic MLE parameter estimation
and physically-constrained sampling.

Example:
    >>> from openad_lib.feedstock.distributions import BetaDistribution
    >>> # Fit from measurements
    >>> biodeg_samples = [0.75, 0.82, 0.78, 0.80, 0.76]
    >>> dist = BetaDistribution.fit(biodeg_samples)
    >>> # Sample with constraints
    >>> samples = dist.sample(100)
    >>> print(f"Mean: {dist.mean():.3f}, Std: {dist.std():.3f}")
"""

import numpy as np
from scipy import stats
from typing import Optional, Union, List
from abc import ABC, abstractmethod
import warnings


class Distribution(ABC):
    """
    Base class for probability distributions.
    
    Provides common interface for all distribution types used in
    feedstock uncertainty quantification.
    """
    
    @abstractmethod
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the distribution.
        
        Args:
            n: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            Array of samples
        """
        pass
    
    @abstractmethod
    def mean(self) -> float:
        """Calculate distribution mean."""
        pass
    
    @abstractmethod
    def std(self) -> float:
        """Calculate distribution standard deviation."""
        pass
    
    @abstractmethod
    def quantile(self, q: Union[float, List[float]]) -> Union[float, np.ndarray]:
        """
        Calculate quantile(s).
        
        Args:
            q: Quantile(s) to compute (0-1)
            
        Returns:
            Quantile value(s)
        """
        pass
    
    @classmethod
    @abstractmethod
    def fit(cls, samples: Union[List[float], np.ndarray]) -> 'Distribution':
        """
        Fit distribution parameters using Maximum Likelihood Estimation.
        
        Args:
            samples: Observed data samples
            
        Returns:
            Fitted distribution instance
        """
        pass


class BetaDistribution(Distribution):
    """
    Beta distribution for bounded fractions (0-1).
    
    Used for parameters that must be between 0 and 1, such as:
    - Biodegradability coefficients (f_d)
    - VS/TS ratios
    - Soluble/Total COD fractions
    
    Attributes:
        alpha: Shape parameter α (> 0)
        beta: Shape parameter β (> 0)
    
    Example:
        >>> # Biodegradability measurements
        >>> f_d_samples = [0.75, 0.82, 0.78, 0.80, 0.76, 0.79]
        >>> dist = BetaDistribution.fit(f_d_samples)
        >>> print(f"Mean: {dist.mean():.3f}")
        >>> samples = dist.sample(1000)
        >>> assert all(0 <= s <= 1 for s in samples)
    """
    
    def __init__(self, alpha: float, beta: float):
        """
        Initialize Beta distribution.
        
        Args:
            alpha: Shape parameter α (> 0)
            beta: Shape parameter β (> 0)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Alpha and beta must be positive, got α={alpha}, β={beta}")
        
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.beta(alpha, beta)
    
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples constrained to [0, 1]."""
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = self._dist.rvs(size=n)
        # Ensure strict bounds (should already be satisfied, but enforce)
        samples = np.clip(samples, 0.0, 1.0)
        return samples
    
    def mean(self) -> float:
        """Calculate mean: α / (α + β)"""
        return self._dist.mean()
    
    def std(self) -> float:
        """Calculate standard deviation."""
        return self._dist.std()
    
    def quantile(self, q: Union[float, List[float]]) -> Union[float, np.ndarray]:
        """Calculate quantile(s)."""
        return self._dist.ppf(q)
    
    @classmethod
    def fit(cls, samples: Union[List[float], np.ndarray]) -> 'BetaDistribution':
        """
        Fit Beta distribution using MLE.
        
        Args:
            samples: Observed data (must be in [0, 1])
            
        Returns:
            Fitted BetaDistribution
            
        Raises:
            ValueError: If samples are outside [0, 1]
        """
        samples = np.asarray(samples)
        
        # Validate bounds
        if np.any(samples < 0) or np.any(samples > 1):
            raise ValueError("Beta distribution requires samples in [0, 1]")
        
        # Remove exact 0s and 1s (causes issues with MLE)
        epsilon = 1e-6
        samples = np.clip(samples, epsilon, 1 - epsilon)
        
        # MLE fitting
        alpha, beta, loc, scale = stats.beta.fit(samples, floc=0, fscale=1)
        
        return cls(alpha=alpha, beta=beta)
    
    def __repr__(self) -> str:
        return f"BetaDistribution(α={self.alpha:.3f}, β={self.beta:.3f})"


class LogNormalDistribution(Distribution):
    """
    Log-normal distribution for strictly positive quantities.
    
    Used for parameters that must be positive, such as:
    - Total Solids (TS)
    - Volatile Solids (VS)
    - COD concentrations
    - BMP values
    
    Attributes:
        mu: Location parameter (mean of log(X))
        sigma: Scale parameter (std of log(X))
    
    Example:
        >>> # TS measurements [kg/m³]
        >>> ts_samples = [310, 315, 308, 312, 318, 305]
        >>> dist = LogNormalDistribution.fit(ts_samples)
        >>> samples = dist.sample(1000)
        >>> assert all(s > 0 for s in samples)
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Initialize Log-normal distribution.
        
        Args:
            mu: Location parameter (mean of log(X))
            sigma: Scale parameter (std of log(X), > 0)
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got σ={sigma}")
        
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples (always positive)."""
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = self._dist.rvs(size=n)
        # Ensure positivity (should already be satisfied)
        samples = np.maximum(samples, 1e-10)
        return samples
    
    def mean(self) -> float:
        """Calculate mean: exp(μ + σ²/2)"""
        return self._dist.mean()
    
    def std(self) -> float:
        """Calculate standard deviation."""
        return self._dist.std()
    
    def quantile(self, q: Union[float, List[float]]) -> Union[float, np.ndarray]:
        """Calculate quantile(s)."""
        return self._dist.ppf(q)
    
    @classmethod
    def fit(cls, samples: Union[List[float], np.ndarray]) -> 'LogNormalDistribution':
        """
        Fit Log-normal distribution using MLE.
        
        Args:
            samples: Observed data (must be positive)
            
        Returns:
            Fitted LogNormalDistribution
            
        Raises:
            ValueError: If samples contain non-positive values
        """
        samples = np.asarray(samples)
        
        # Validate positivity
        if np.any(samples <= 0):
            raise ValueError("Log-normal distribution requires positive samples")
        
        # MLE fitting: fit to log-transformed data
        log_samples = np.log(samples)
        mu = np.mean(log_samples)
        sigma = np.std(log_samples, ddof=1)
        
        return cls(mu=mu, sigma=sigma)
    
    def __repr__(self) -> str:
        return f"LogNormalDistribution(μ={self.mu:.3f}, σ={self.sigma:.3f})"


class GammaDistribution(Distribution):
    """
    Gamma distribution for strictly positive quantities.
    
    Alternative to Log-normal for positive parameters. Often preferred
    when data is less skewed or has heavier tails.
    
    Used for:
    - Protein content
    - Lipid content
    - TAN (Total Ammonia Nitrogen)
    
    Attributes:
        shape: Shape parameter k (> 0)
        scale: Scale parameter θ (> 0)
    
    Example:
        >>> # Protein measurements [kg/m³]
        >>> protein_samples = [28, 32, 30, 31, 29, 33]
        >>> dist = GammaDistribution.fit(protein_samples)
        >>> samples = dist.sample(1000)
        >>> assert all(s > 0 for s in samples)
    """
    
    def __init__(self, shape: float, scale: float):
        """
        Initialize Gamma distribution.
        
        Args:
            shape: Shape parameter k (> 0)
            scale: Scale parameter θ (> 0)
        """
        if shape <= 0 or scale <= 0:
            raise ValueError(f"Shape and scale must be positive, got k={shape}, θ={scale}")
        
        self.shape = shape
        self.scale = scale
        self._dist = stats.gamma(a=shape, scale=scale)
    
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples (always positive)."""
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = self._dist.rvs(size=n)
        # Ensure positivity (should already be satisfied)
        samples = np.maximum(samples, 1e-10)
        return samples
    
    def mean(self) -> float:
        """Calculate mean: k * θ"""
        return self._dist.mean()
    
    def std(self) -> float:
        """Calculate standard deviation: √(k * θ²)"""
        return self._dist.std()
    
    def quantile(self, q: Union[float, List[float]]) -> Union[float, np.ndarray]:
        """Calculate quantile(s)."""
        return self._dist.ppf(q)
    
    @classmethod
    def fit(cls, samples: Union[List[float], np.ndarray]) -> 'GammaDistribution':
        """
        Fit Gamma distribution using MLE.
        
        Args:
            samples: Observed data (must be positive)
            
        Returns:
            Fitted GammaDistribution
            
        Raises:
            ValueError: If samples contain non-positive values
        """
        samples = np.asarray(samples)
        
        # Validate positivity
        if np.any(samples <= 0):
            raise ValueError("Gamma distribution requires positive samples")
        
        # MLE fitting
        shape, loc, scale = stats.gamma.fit(samples, floc=0)
        
        return cls(shape=shape, scale=scale)
    
    def __repr__(self) -> str:
        return f"GammaDistribution(k={self.shape:.3f}, θ={self.scale:.3f})"


def assign_distribution(
    parameter_name: str,
    samples: Union[List[float], np.ndarray]
) -> Distribution:
    """
    Automatically assign appropriate distribution type based on parameter.
    
    Assignment logic:
    - Beta: Bounded fractions (biodegradability, VS/TS ratio, etc.)
    - LogNormal: Positive quantities (TS, VS, COD, BMP)
    - Gamma: Alternative for positive quantities (proteins, lipids, TAN)
    
    Args:
        parameter_name: Name of the parameter
        samples: Observed data samples
        
    Returns:
        Fitted distribution of appropriate type
        
    Example:
        >>> samples = [0.75, 0.82, 0.78, 0.80]
        >>> dist = assign_distribution('biodegradability', samples)
        >>> print(type(dist).__name__)
        BetaDistribution
    """
    samples = np.asarray(samples)
    param_lower = parameter_name.lower()
    
    # Beta distribution for bounded fractions
    if any(keyword in param_lower for keyword in [
        'biodegradability', 'f_d', 'vs_fraction', 'fraction', 'ratio'
    ]):
        # Check if samples are in [0, 1]
        if np.all(samples >= 0) and np.all(samples <= 1):
            return BetaDistribution.fit(samples)
    
    # LogNormal for primary positive quantities
    if any(keyword in param_lower for keyword in [
        'ts', 'vs', 'cod', 'bmp', 'solids'
    ]):
        return LogNormalDistribution.fit(samples)
    
    # Gamma for secondary positive quantities
    if any(keyword in param_lower for keyword in [
        'protein', 'lipid', 'tan', 'carbohydrate', 'acetate', 'propionate'
    ]):
        return GammaDistribution.fit(samples)
    
    # Default: LogNormal for positive data, otherwise warn
    if np.all(samples > 0):
        warnings.warn(
            f"Parameter '{parameter_name}' not recognized, defaulting to LogNormal",
            UserWarning
        )
        return LogNormalDistribution.fit(samples)
    else:
        raise ValueError(
            f"Cannot assign distribution for '{parameter_name}' with non-positive samples"
        )
