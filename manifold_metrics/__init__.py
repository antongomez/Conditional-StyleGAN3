"""
Utilities to compute manifold metrics for multispectral images.
"""

from .manifold_metrics import calculate_fid, compute_precision_recall

__version__ = "1.0.0"
__author__ = "Antón Gómez López"

__all__ = [
    "calculate_fid",
    "compute_precision_recall",
]
