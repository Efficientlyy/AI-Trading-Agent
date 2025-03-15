"""Factory for creating market regime detection algorithms."""

from typing import Dict, Any, Optional, List, Type, Union

from .base_detector import BaseRegimeDetector
from .volatility_detector import VolatilityRegimeDetector
from .momentum_detector import MomentumRegimeDetector
from .hmm_detector import HMMRegimeDetector
from .trend_detector import TrendRegimeDetector
from .ensemble_detector import EnsembleRegimeDetector


class RegimeDetectorFactory:
    """
    Factory for creating market regime detection algorithms.
    
    This factory provides a simple interface for creating and using different
    regime detection methods, with sensible defaults and configuration options.
    """
    
    # Registry of available detectors
    _registry: Dict[str, Type[BaseRegimeDetector]] = {
        'volatility': VolatilityRegimeDetector,
        'momentum': MomentumRegimeDetector,
        'hmm': HMMRegimeDetector,
        'trend': TrendRegimeDetector,
        'ensemble': EnsembleRegimeDetector,
    }
    
    @classmethod
    def register(cls, name: str, detector_class: Type[BaseRegimeDetector]) -> None:
        """
        Register a new detector class.
        
        Args:
            name: Name of the detector
            detector_class: Detector class to register
        """
        cls._registry[name.lower()] = detector_class
    
    @classmethod
    def create(cls, method: str, **kwargs) -> BaseRegimeDetector:
        """
        Create a detector instance.
        
        Args:
            method: Name of the detector method
            **kwargs: Additional parameters for the detector
            
        Returns:
            Detector instance
            
        Raises:
            ValueError: If the method is not registered
        """
        method = method.lower()
        if method not in cls._registry:
            raise ValueError(f"Unknown regime detection method: {method}")
        
        detector_class = cls._registry[method]
        return detector_class(**kwargs)
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """
        Get a list of available detector methods.
        
        Returns:
            List of method names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create_all(cls, methods: Optional[List[str]] = None, **kwargs) -> Dict[str, BaseRegimeDetector]:
        """
        Create multiple detector instances.
        
        Args:
            methods: List of method names (default: all registered methods)
            **kwargs: Additional parameters for all detectors
            
        Returns:
            Dictionary of detector instances
        """
        if methods is None:
            methods = cls.get_available_methods()
        
        detectors = {}
        for method in methods:
            try:
                detectors[method] = cls.create(method, **kwargs)
            except ValueError:
                # Skip unknown methods
                continue
        
        return detectors


# Add the factory to the package exports
__all__ = ['RegimeDetectorFactory']

# README location for documentation
README = 'README.md' 