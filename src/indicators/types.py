"""Type definitions for technical indicators."""

from typing import TypedDict, List, Dict, Union, TypeVar, Protocol
import numpy as np
from numpy.typing import NDArray


class HeikinAshiCandles(TypedDict):
    """Heikin Ashi candle data."""
    open: Union[List[float], NDArray[np.float64]]
    high: Union[List[float], NDArray[np.float64]]
    low: Union[List[float], NDArray[np.float64]]
    close: Union[List[float], NDArray[np.float64]]


class KeltnerChannels(TypedDict):
    """Keltner channel data."""
    upper: Union[List[float], NDArray[np.float64]]
    middle: Union[List[float], NDArray[np.float64]]
    lower: Union[List[float], NDArray[np.float64]]


class MarketRegime(TypedDict):
    """Market regime data."""
    trending: NDArray[np.bool_]
    ranging: NDArray[np.bool_]


# Type aliases for common indicator inputs/outputs
PriceData = Union[List[float], NDArray[np.float64]]
VolumeData = Union[List[float], NDArray[np.float64]]
IndicatorOutput = Union[List[float], NDArray[np.float64]] 