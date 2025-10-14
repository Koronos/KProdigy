# Changelog

All notable changes to K-Prodigy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-13

### Added
- Multi-tensor (foreach) operations for ~21% GPU speedup
- Independent D estimation per parameter group for multi-component models
- Adaptive bias correction for improved stability
- Automatic CUDA detection with CPU fallback
- Comprehensive documentation and examples

### Fixed
- EMA scaling issue in original Prodigy (removed D scaling from gradient accumulation)
- Memory format preservation in tensor initialization
- Optimized parameter initialization checks

### Performance
- **21% faster** than original Prodigy on average (GPU)
- **11.5% faster** on SDXL-style UNet models
- **30.3% faster** on large MLP models
- Equal or better convergence quality

### Technical Improvements
- Scale-invariant Adam denominator using `torch.maximum`
- Optimized D calculation with batched GPU operations
- Better numerical stability throughout

---

## Project Origin

K-Prodigy is based on the [Prodigy optimizer](https://github.com/konstmish/prodigy) by Konstantin Mishchenko and Aaron Defazio:

> Mishchenko, K., & Defazio, A. (2023). *Prodigy: An Expeditiously Adaptive Parameter-Free Learner*. arXiv preprint arXiv:2306.06101.

[0.1.0]: https://github.com/Koronos/KProdigy/releases/tag/v0.1.0

