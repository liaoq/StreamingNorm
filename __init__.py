"""
Streaming Normalization (snorm) package

This package implements streaming normalization for deep networks.
"""

from .streaming import Streaming, StreamingFunction
from .streaming_norm import StreamingNorm1d, StreamingNorm2d

__all__ = ['Streaming', 'StreamingFunction', 'StreamingNorm1d', 'StreamingNorm2d']

