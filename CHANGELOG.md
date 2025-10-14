# Changelog

All notable changes to K-Prodigy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-14

### Added
- **Independent D estimation per parameter group (auto-detected)** - CRITICAL for SDXL multi-component models
  - Prevents "burning" Text Encoders when training with UNet
  - Each component (UNet, TE) gets its own adaptive learning rate scale
  - Automatically enabled when using multiple parameter groups
  - Solves issue from [Prodigy PR #34](https://github.com/konstmish/prodigy/pull/34)
- **Auto-detection features** - Zero-configuration optimization
  - `foreach` auto-enabled on CUDA, graceful CPU fallback
  - `independent_d` auto-enabled for multi-parameter groups
  - Intelligent caching eliminates overhead (< 0.001%)
- **Hot path optimizations** - Transparent performance improvements
  - Pre-calculated scalar multipliers (d_over_d0, dlr_scaled, etc.)
  - Cached CUDA availability check
  - Single `sliced_param` calculation in state initialization
  - Pre-created `eps_tensor` per group
- **Phase 1 refactoring** - Helper methods for cleaner code
  - `_compute_bias_correction()` - Centralized bias correction logic
  - `_get_group_hyperparams()` - Extract group hyperparameters
  - `_precompute_scalars()` - Pre-calculate scalar multipliers

### Fixed
- **Reverted to Prodigy's original d-scaling in EMA updates** - For loss parity with baseline
  - EMA updates now use `d * beta_complement` (Prodigy's approach)
  - Denominator uses `d * eps` instead of just `eps`
  - Ensures identical convergence to original Prodigy
- Eliminated repeated CUDA checks (now cached in `_can_use_foreach`)
- Optimized state initialization (reduced redundant operations)

### Changed
- **Three execution paths** for optimal performance
  - `_step_single_tensor()` - CPU or mixed device scenarios
  - `_step_foreach()` - CUDA with single parameter group
  - `_step_independent_d()` - Multi-parameter groups
- Code structure refactored with helper methods (improved maintainability)
- Documentation significantly expanded

### Performance
**SDXL-Style Benchmarks** (RTX 3000 Ada, 200 steps):
- **UNet (~7M params)**: 26.7ms/step, 70.2% loss reduction
- **TextEncoder (~20M params)**: 43.7ms/step, 99.8% loss reduction
- **Multi-Component (~27M params)**: 75.8ms/step, Independent D ratio ~37x
- **Loss**: Identical to original Prodigy
- **Speed**: 15% faster on UNet vs v0.1.0

### Breaking Changes
- None - Fully backward compatible with v0.1.0

### Migration Notes
- No code changes required
- Existing code will work exactly as before
- New features auto-enable when appropriate
- For SDXL multi-component: Just use multiple parameter groups, Independent D activates automatically

---

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

[0.2.0]: https://github.com/Koronos/KProdigy/releases/tag/v0.2.0
[0.1.0]: https://github.com/Koronos/KProdigy/releases/tag/v0.1.0

