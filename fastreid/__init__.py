# encoding: utf-8
"""
FastReID: A PyTorch library for person re-identification, vehicle re-identification and general instance retrieval.

FastReID provides state-of-the-art algorithms and tools for re-identification tasks.
It supports person ReID, vehicle ReID, and general instance retrieval with modern PyTorch implementations.

Basic Usage:
    >>> import fastreid
    >>> from fastreid.config import get_cfg
    >>> from fastreid.modeling import build_model

    >>> # Load configuration
    >>> cfg = get_cfg()
    >>> cfg.merge_from_file("path/to/config.yml")
    >>>
    >>> # Build model
    >>> model = build_model(cfg)
    >>>
    >>> # Extract features
    >>> features = model(images)

Components:
    - config: Configuration management
    - data: Data loading and preprocessing
    - engine: Training and evaluation
    - evaluation: Metrics and evaluation tools
    - layers: Custom neural network layers
    - modeling: Models, backbones, and heads
    - solver: Optimizers and schedulers
    - utils: Utility functions

@author:  Xingyu Liao
@contact: sherlockliao01@gmail.com
"""

__version__ = "1.3.0"
__author__ = "Xingyu Liao"
__email__ = "sherlockliao01@gmail.com"

# Core imports
from . import config
from . import data
from . import engine
from . import evaluation
from . import layers
from . import modeling
from . import solver
from . import utils

# Main API functions
from .config import get_cfg
from .modeling import build_model

# Version info
def get_version():
    """Get the current version of FastReID."""
    return __version__

# Public API
__all__ = [
    "__version__",
    "get_version",
    "config",
    "data",
    "engine",
    "evaluation",
    "layers",
    "modeling",
    "solver",
    "utils",
    "get_cfg",
    "build_model",
]
