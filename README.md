# K-Prodigy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-EE4C2C.svg)](https://pytorch.org/)

An enhanced version of the [Prodigy optimizer](https://github.com/konstmish/prodigy) with ~21% GPU speedup and multi-component model support.

## Key Features

- ⚡ **21% faster** on GPU via multi-tensor operations
- 🎨 **Multi-component support** with independent D estimation (perfect for SDXL!)
- 🎯 **Adaptive bias correction** for improved stability
- 🔧 **Drop-in replacement** for Prodigy - same API, better performance

## Installation

### Direct from GitHub (Recommended)

```bash
pip install git+https://github.com/Koronos/KProdigy.git
```

### Development Installation

```bash
git clone https://github.com/Koronos/KProdigy.git
cd K-Prodigy
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from kprodigy import KProdigy

model = torch.nn.Linear(10, 1)
optimizer = KProdigy(model.parameters(), lr=1.0)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(model(data), target)
    loss.backward()
    optimizer.step()
```

### For Multi-Component Models (SDXL, etc.)

```python
# Prevents "burning" the Text Encoder
optimizer = KProdigy([
    {'params': unet.parameters(), 'lr': 1.0},
    {'params': text_encoder.parameters(), 'lr': 0.5}
], independent_d=True)  # KEY: Separate D estimate per component
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1.0` | Learning rate multiplier (typically leave at 1.0) |
| `betas` | `(0.9, 0.999)` | Coefficients for gradient moving averages |
| `weight_decay` | `0.0` | Weight decay (L2 penalty) |
| `decouple` | `True` | Use AdamW-style decoupled weight decay |
| `use_bias_correction` | `False` | Enable Adam-style bias correction |
| `foreach` | `True` | Enable multi-tensor ops for GPU speedup |
| `independent_d` | `False` | Calculate separate D per parameter group |

## Performance Comparison

Tested on RTX 3000 Ada with SDXL-style models:

| Metric | Original Prodigy | **K-Prodigy** | Improvement |
|--------|-----------------|-------------|-------------|
| **UNet Training Time** | 39.42s | **34.89s** | **+11.5% faster** |
| **MLP Training Time** | 18.42s | **12.83s** | **+30.3% faster** |
| **Overall Speed** | Baseline | **+21.2% faster** | ⚡ |
| **Convergence** | Baseline | Equal/Better | ✅ |

## When to Use K-Prodigy

### ✅ Recommended For:

- Training diffusion models (Stable Diffusion, SDXL, etc.)
- Multi-component models with different learning rate needs
- GPU training where speed matters
- When you want Prodigy's benefits with better performance

### 🤔 Consider Original Prodigy If:

- You need the exact reference implementation
- You're training on CPU exclusively

## Configuration Examples

### Diffusion Models (SDXL)

```python
optimizer = KProdigy(
    model.parameters(),
    lr=1.0,
    betas=(0.9, 0.99),
    weight_decay=0.01,
    use_bias_correction=True
)
```

### Multi-Component (UNet + Text Encoders)

```python
optimizer = KProdigy([
    {'params': unet.parameters(), 'lr': 1.0, 'weight_decay': 0.01},
    {'params': text_encoder.parameters(), 'lr': 0.5, 'weight_decay': 0.001}
], 
independent_d=True,  # CRITICAL for multi-component
betas=(0.9, 0.99),
use_bias_correction=True
)
```

## Technical Details

### Enhancements Over Original Prodigy

1. **Multi-tensor operations**: Uses `torch._foreach_*` for batched GPU operations
2. **Independent D estimation**: Separate learning rate adaptation per parameter group
3. **Adaptive bias correction**: Scales bias correction by D/D0 for improved stability
4. **Optimized denominator calculation**: Better numerical stability with `torch.maximum`

### Why Independent D?

In multi-component models like SDXL:
- UNet (~2.6B params) needs aggressive learning rates
- Text Encoder (~300M params) is more sensitive
- Without independent D: UNet's gradients dominate → Text Encoder gets "burned"
- With independent D: Each component adapts at its own pace ✅

## Citation

If you use K-Prodigy in your research, please cite both K-Prodigy and the original Prodigy paper:

```bibtex
@software{kprodigy2025,
  title={K-Prodigy: Enhanced Prodigy Optimizer with GPU Acceleration},
  author={Koronos},
  year={2025},
  url={https://github.com/Koronos/KProdigy}
}

@article{mishchenko2023prodigy,
  title={Prodigy: An Expeditiously Adaptive Parameter-Free Learner},
  author={Mishchenko, Konstantin and Defazio, Aaron},
  journal={arXiv preprint arXiv:2306.06101},
  year={2023}
}
```

## References

- [Original Prodigy Paper](https://arxiv.org/abs/2306.06101)
- [Prodigy Repository](https://github.com/konstmish/prodigy)
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)

## License

MIT License - see [LICENSE](LICENSE) for details.

Based on [Prodigy optimizer](https://github.com/konstmish/prodigy) by Konstantin Mishchenko and Aaron Defazio.

