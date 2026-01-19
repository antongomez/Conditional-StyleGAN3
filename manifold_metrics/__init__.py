"""
Utilities to compute manifold metrics for multispectral images.
"""

from .cnn2d_residual import HYPERPARAMS, CNN2D_Residual, DatasetMock
from .manifold_metrics import calculate_fid, compute_precision_recall
from .manifold_utils import get_synthetic_features, get_train_features

__version__ = "1.0.0"
__author__ = "Antón Gómez López"

__all__ = [
    "calculate_fid",
    "compute_precision_recall",
    "CNN2D_Residual",
    "DatasetMock",
    "HYPERPARAMS",
    "get_synthetic_features",
    "get_train_features",
]
