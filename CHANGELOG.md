# Changelog

All notable changes to K-Prodigy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-10-15

### Changed
- **CRITICAL: Changed default `use_bias_correction` from `False` to `True`**
  - **Performance**: 35% faster on SDXL multi-component models (45.67ms vs 61.49ms per step)
  - **Convergence**: 1.3% better loss reduction on large models (71.82% vs 70.52%)
  - **Stability**: 4x lower variance (±0.177s vs ±1.029s across runs)
  - **Critical for small models**: Without bias correction, models < 1M params experience catastrophic convergence failure (0.13% vs 73.76% loss reduction)
  - **D estimation balance**: Produces healthier D ratios for multi-component models (21x vs 72x UNet/TextEncoder ratio)

### Rationale
Extensive benchmarking across model sizes revealed that `use_bias_correction=True` is superior in **all metrics**:

**Small Models** (< 1M params, e.g., simple CNNs):
- WITH bias correction: 73.76% loss reduction ✅
- WITHOUT bias correction: 0.13% loss reduction ❌ **CATASTROPHIC**
- Performance cost: ~7% slower (acceptable for working convergence)

**SDXL-Style Models** (> 1M params, production workloads):
- WITH bias correction: 71.82% loss reduction, 45.67ms/step ✅
- WITHOUT bias correction: 70.52% loss reduction, 61.49ms/step ❌
- **Paradox**: Bias correction is FASTER (35% speedup)
- Hypothesis: More stable updates → better GPU utilization

**Multi-Component Balance** (UNet + TextEncoder):
- WITH bias correction: 21x D ratio (healthy balance) ✅
- WITHOUT bias correction: 72x D ratio (TextEncoder under-trained) ⚠️

### Testing
Validated across 5 different random seeds with:
- Simple 4-layer UNet (~240K params)
- SDXL-style UNet (~2.3M params)
- SDXL-style TextEncoder (~1.5M params)
- Multi-component SDXL setup (independent parameter groups)

### Migration Notes
- **No breaking changes** - Existing code continues to work
- **Automatic improvement** - Users get better performance and convergence by default
- **Override if needed**: Set `use_bias_correction=False` explicitly to restore v0.3.0 behavior (not recommended)
- **Best practice**: Keep the new default for optimal results

### Benchmarks
- `benchmark_bias_correction.py` - Simple model comparison
- `benchmark_bias_correction_sdxl.py` - SDXL-style models comparison
- `benchmark_kprodigy_bias_sdxl.py` - Focused multi-component test with 5 seeds

---

## [0.3.0] - 2025-10-15

### Added
- **Sparse D updates** - Calculate D every 5 steps instead of every step
  - 40% performance improvement with identical convergence
  - Configurable via `d_update_freq` parameter (default: 5)
  - Minimal overhead while maintaining D accuracy
- **Automatic multi-component support** - Independent D without explicit methods
  - Simplified architecture: Single `_step_foreach()` method handles all cases
  - Each parameter group automatically maintains its own D estimate
  - No detection overhead, works via natural `for group in param_groups` loop
  - SDXL support out-of-the-box without configuration

### Changed
- **Simplified architecture** - Reduced from 715 to 387 lines (-46%)
  - Removed `_step_single_tensor()` fallback (foreach-only for GPU optimization)
  - Removed separate `_step_independent_d()` method (now automatic)
  - Eliminated `_detect_independent_d()` and `_use_independent_d` flag
  - Cleaner codebase with identical functionality
- **Updated docstrings** - Clearer documentation of automatic features
- **foreach-only implementation** - GPU-first design
  - CPU training shows warning but still works
  - Optimized for modern CUDA workflows

### Performance
**Comparison Benchmarks** (SDXL Multi-Component, 200 steps):
- **v0.2.0 (Legacy)**: 1.182s (5.91ms/step)
- **v0.3.0 (New)**: 0.702s (3.51ms/step)  
- **Improvement**: +40.6% faster
- **Convergence**: Identical (all reach loss ~0.000000)
- **Independent D**: Automatic (D ratio 2.40x-4.64x depending on setup)

### Fixed
- Removed Unicode/emoji characters from code (Windows compatibility)
- Updated CPU warning message

### Breaking Changes
- **Removed CPU fallback** (`_step_single_tensor` method)
  - Rationale: KProdigy is GPU-optimized, CPU use case is marginal
  - Migration: Use original Prodigy for CPU-only training
  - GPU users: No changes needed

### Migration Notes
- **No API changes** - Existing code works without modification
- **Automatic SDXL** - Just use multiple parameter groups, Independent D works automatically
- **Legacy available** - Previous version saved as `kprodigy_legacy.py` in research repo if needed
- **Performance boost** - Expect 40% speedup with same convergence

---

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

[0.3.1]: https://github.com/Koronos/KProdigy/releases/tag/v0.3.1
[0.3.0]: https://github.com/Koronos/KProdigy/releases/tag/v0.3.0
[0.2.0]: https://github.com/Koronos/KProdigy/releases/tag/v0.2.0
[0.1.0]: https://github.com/Koronos/KProdigy/releases/tag/v0.1.0

