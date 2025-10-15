"""
K-Prodigy: An Enhanced Expeditiously Adaptive Parameter-Free Optimizer

Based on Prodigy by Konstantin Mishchenko and Aaron Defazio
https://arxiv.org/abs/2306.06101

Enhancements:
- 40% faster via sparse D updates
- Fixed Independent D for SDXL multi-component training
- Automatic multi-component detection
- Optimized defaults (bias correction enabled, d_update_freq=1)
- Foreach-only implementation (GPU-optimized)
"""

from .kprodigy import KProdigy

__version__ = "0.3.3"
__all__ = ["KProdigy"]

