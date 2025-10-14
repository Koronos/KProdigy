"""
K-Prodigy: An Enhanced Expeditiously Adaptive Parameter-Free Optimizer

Based on Prodigy by Konstantin Mishchenko and Aaron Defazio
https://arxiv.org/abs/2306.06101

Enhancements:
- ~21% faster on GPU via multi-tensor operations
- Independent D estimation for multi-component models
- Improved stability with adaptive bias correction
"""

from .kprodigy import KProdigy

__version__ = "0.1.0"
__all__ = ["KProdigy"]

