"""Isolation-based anomaly detection, batch and streaming, in plain NumPy."""

from .detector import StreamDetector
from .hstree import HalfSpaceTrees
from .iforest import IsolationForest

__all__ = ["IsolationForest", "HalfSpaceTrees", "StreamDetector"]
__version__ = "0.1.0"
