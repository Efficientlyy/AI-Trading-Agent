"""
Alternative Data Integration Module for AI Trading Agent.

This module provides interfaces and implementations for various alternative data sources
to enhance trading signals and market insights beyond traditional market data.

Available data sources:
- Satellite imagery analysis
- Social media sentiment
- Supply chain and logistics data
"""

from .base import AlternativeDataSource, AlternativeDataConfig
from .satellite_imagery import SatelliteImageryAnalyzer
from .social_media import SocialMediaSentimentAnalyzer
from .supply_chain import SupplyChainDataAnalyzer

__all__ = [
    'AlternativeDataSource',
    'AlternativeDataConfig',
    'SatelliteImageryAnalyzer',
    'SocialMediaSentimentAnalyzer',
    'SupplyChainDataAnalyzer',
]
