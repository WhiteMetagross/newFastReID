# encoding: utf-8
"""
FastReID Command Line Tools

This module provides command-line interfaces for common FastReID operations.
"""

from .train import main as train_main
from .test import main as test_main  
from .demo import main as demo_main

__all__ = [
    "train_main",
    "test_main", 
    "demo_main",
]
