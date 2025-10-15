"""
K-Prodigy: An Enhanced Expeditiously Adaptive Parameter-Free Optimizer

Based on Prodigy by Konstantin Mishchenko and Aaron Defazio
https://arxiv.org/abs/2306.06101

Enhancements:
- 40% faster via sparse D updates (every 5 steps)
- Automatic multi-component support (SDXL)
- Simplified architecture (387 lines vs 715)
- Foreach-only implementation (GPU-optimized)
"""

from .kprodigy import KProdigy

__version__ = "0.3.0"
__all__ = ["KProdigy"]

