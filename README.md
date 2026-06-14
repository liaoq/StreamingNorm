# Streaming Normalization (snorm)

**Python/PyTorch implementation of Streaming Normalization** - a normalization technique that simultaneously solves two major limitations of Batch Normalization: (1) online learning and (2) recurrent learning.

This is a quick Python implementation for the paper:

> **Streaming Normalization: Towards Simpler and More Biologically-plausible Normalizations for Online and Recurrent Learning**  
> Qianli Liao, Kenji Kawaguchi, Tomaso Poggio  
> CBMM Memo No. 057, October 19, 2016  
> [arXiv:1610.06160](https://arxiv.org/abs/1610.06160)

Note that the original experiments in the paper were done with Matlab. This python implementation may differ a bit in details but the algorithmic core should be similar enough.

The code is vibe-coded with AI so let me know if you have any questions or you find any issue.

## Overview

Streaming Normalization maintains normalization statistics in an **online fashion** from all previously seen training samples (and all timesteps if recurrent). Unlike Batch Normalization, it works out of the box in all learning scenarios:

- ✅ **Online learning** (pure online or small mini-batches)
- ✅ **Recurrent learning** (RNNs, GRUs, layer-shared networks)
- ✅ **Feedforward networks** (fully-connected and convolutional)
- ✅ **Mixed architectures** (recurrent and convolutional)

### Key Advantages

| Feature | BatchNorm | LayerNorm | **StreamingNorm** |
|---------|-----------|-----------|-------------------|
| Feedforward & FC | ✅ | ✅ | ✅ |
| Feedforward & Conv | ✅ | ⚠️ | ✅ |
| Recurrent & FC | ❌ | ✅ | ✅ |
| Recurrent & Conv | ❌ | ⚠️ | ✅ |
| Online Learning | ❌ | ✅ | ✅ |
| Small Batch | ⚠️ | ✅ | ✅ |
| All Combined | ❌ | ❌ | ✅ |

## Installation

```bash
# Clone the repository
git clone https://github.com/REPO_NAME/snorm.git

# Install dependencies (PyTorch)
pip install torch
```

## Quick Start

### Replace BatchNorm with StreamingNorm

```python
import torch.nn as nn
from snorm import StreamingNorm1d, StreamingNorm2d

# For MLPs (1D)
# Before: norm = nn.BatchNorm1d(512)
norm = StreamingNorm1d(num_features=512)

# For CNNs (2D)  
# Before: norm = nn.BatchNorm2d(64)
norm = StreamingNorm2d(num_features=64)

# Use it exactly like BatchNorm
x = norm(x)
```

### Complete Example: MLP

```python
import torch
import torch.nn as nn
from snorm import StreamingNorm1d

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = StreamingNorm1d(num_features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = StreamingNorm1d(num_features=hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# Use it
model = MLP()
x = torch.randn(32, 784)  # batch_size=32
output = model(x)
```

### Complete Example: CNN

```python
import torch
import torch.nn as nn
from snorm import StreamingNorm2d

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.norm1 = StreamingNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm2 = StreamingNorm2d(num_features=128)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = torch.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Use it
model = CNN()
x = torch.randn(32, 3, 32, 32)  # batch_size=32, CIFAR-10
output = model(x)
```

## How It Works

### Theoretical Foundation

Streaming Normalization maintains normalization statistics in an online fashion using:

1. **Short-term Statistics** (`ŝ_short`): Exact average of normalization statistics since the **last weight update**
2. **Long-term Statistics** (`ŝ_long`): Exponential average of short-term statistics since the **beginning of training**

The final normalization statistics used are:
```
ŝ = α₁·ŝ_long + α₂·ŝ_short + α₃·s_current
```

Similarly, **Streaming Gradients** are maintained for the gradients of normalization statistics:
```
ĝ = β₁·ĝ_long + β₂·ĝ_short + β₃·g_current
```

**Note:** Our implementation extends the paper by including α₃ (current input) in the forward pass, making it symmetric with the backward pass (β₃). The paper used α₁ + α₂ = 1 with α₃ = 0, but our implementation allows α₃ ≠ 0 for more flexibility.

### Weight Update Detection

Streaming Normalization automatically detects when network weights are updated using a `weight_update_detector` parameter. When a weight update occurs:

1. Long-term statistics are updated: `ŝ_long = κ₁·ŝ_long + κ₂·ŝ_short`
2. Short-term statistics are cleared
3. The cycle repeats

**You don't need to manually set anything** - the detector is handled automatically by the optimizer.

## API Reference

### StreamingNorm1d

1D streaming normalization for MLPs and fully-connected layers.

```python
StreamingNorm1d(
    num_features: int,
    eps: float = 1e-5,
    affine: bool = True,
    decay_factor_f: float = 0.7,
    decay_factor_b: float = 0.7,
    alpha: list = [0.5, 0.5, 0.0],
    beta: list = [0.6, 0.0, 0.4],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

**Parameters:**
- `num_features`: Number of features/channels (same as BatchNorm1d)
- `eps`: Small value to avoid division by zero (default: 1e-5)
- `affine`: If True, apply learnable affine transformation (default: True)
- `decay_factor_f`: Decay factor κ₁ for forward statistics (default: 0.7)
- `decay_factor_b`: Decay factor κ₃ for backward statistics (default: 0.7)
- `alpha`: Weight coefficients [α₁, α₂, α₃] for forward output [long_term, short_term, current] (default: [0.5, 0.5, 0.0])
- `beta`: Weight coefficients [β₁, β₂, β₃] for backward output [long_term, short_term, current] (default: [0.6, 0.0, 0.4])

**Input:** `(N, C)` or `(N, C, L)` where N is batch size, C is number of features, L is sequence length

**Output:** Same shape as input

### StreamingNorm2d

2D streaming normalization for CNNs and convolutional layers.

```python
StreamingNorm2d(
    num_features: int,
    eps: float = 1e-5,
    affine: bool = True,
    decay_factor_f: float = 0.7,
    decay_factor_b: float = 0.7,
    alpha: list = [0.5, 0.5, 0.0],
    beta: list = [0.6, 0.0, 0.4],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

**Parameters:** Same as `StreamingNorm1d`

**Input:** `(N, C, H, W)` where N is batch size, C is number of channels, H and W are spatial dimensions

**Output:** Same shape as input

## Hyperparameters

### Decay Factors (κ₁, κ₂, κ₃, κ₄)

Control how quickly statistics adapt:

- **Higher values** (e.g., 0.9): Slower adaptation, more stable
- **Lower values** (e.g., 0.5): Faster adaptation, more responsive

**Default:** 0.7 for both forward and backward (balanced)

**Note:** In our implementation, `decay_factor_f` corresponds to κ₁ (forward), and `decay_factor_b` corresponds to κ₃ (backward). We set κ₂ = 1 - κ₁ and κ₄ = 1 - κ₃.

### Alpha Coefficients (α₁, α₂, α₃)

Control the mixing of long-term, short-term, and current statistics for **forward pass**:

- `alpha[0]` (α₁): Weight for long-term running average
- `alpha[1]` (α₂): Weight for short-term average (since last weight update)
- `alpha[2]` (α₃): Weight for current input

**Default:** `[0.5, 0.5, 0.0]` (equal mix of long-term and short-term, no current)

**Note:** Our implementation extends the paper by including α₃, making it symmetric with β₃ in the backward pass. The paper used α₁ + α₂ = 1 with α₃ = 0, but we allow α₃ ≠ 0 for more flexibility.

### Beta Coefficients (β₁, β₂, β₃)

Control the mixing of long-term, short-term, and current gradients for **backward pass**:

- `beta[0]` (β₁): Weight for long-term gradient average
- `beta[1]` (β₂): Weight for short-term gradient average
- `beta[2]` (β₃): Weight for current gradient

**Default:** `[0.6, 0.0, 0.4]` (mostly long-term with some current gradient)

### Parameter Choices by Architecture

The following table shows recommended parameter settings for different architectures based on the paper's experiments:

| Architecture | α₁ | α₂ | α₃ | β₁ | β₂ | β₃ | κ₁/κ₃ | Notes |
|--------------|----|----|----|----|----|----|-------|-------|
| **Feedforward FC** | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 | 0.0 | 0.7 | Equal mix of long/short-term |
| **Feedforward FC** (alt) | 0.7 | 0.3 | 0.0 | 0.7 | 0.0 | 0.3 | 0.7 | More weight on long-term |
| **Feedforward Conv** | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 | 0.0 | 0.7 | Same as FC |
| **Recurrent FC** | 0.7 | 0.3 | 0.0 | 0.7 | 0.0 | 0.3 | 0.7 | More stable for RNNs |
| **Recurrent Conv** | 0.7 | 0.3 | 0.0 | 0.7 | 0.0 | 0.3 | 0.7 | Best for layer-shared ResNet |
| **Online Learning** | 0.7 | 0.3 | 0.0 | 0.7 | 0.0 | 0.3 | 0.7 | More stable with small batches |

### Guidance on Choosing Parameters

#### For Feedforward Networks

**Standard setting (recommended for most cases):**
```python
alpha=[0.5, 0.5, 0.0]  # Equal mix of long-term and short-term
beta=[0.5, 0.5, 0.0]   # Equal mix of long-term and short-term gradients
decay_factor_f=0.7     # Balanced adaptation
decay_factor_b=0.7
```

**More stable setting (for difficult training):**
```python
alpha=[0.7, 0.3, 0.0]   # More weight on long-term (more stable)
beta=[0.7, 0.0, 0.3]    # Mix of long-term and current gradients
decay_factor_f=0.7
decay_factor_b=0.7
```

#### For Recurrent Networks

**Recommended setting (from paper):**
```python
alpha=[0.7, 0.3, 0.0]   # More weight on long-term for stability
beta=[0.7, 0.0, 0.3]    # Mix of long-term and current gradients
decay_factor_f=0.7
decay_factor_b=0.7
```

**Why this works:** Recurrent networks benefit from more stable statistics. Using more long-term statistics (α₁=0.7) helps maintain consistent normalization across timesteps.

#### For Online Learning (batch size = 1)

**Recommended setting:**
```python
alpha=[0.7, 0.3, 0.0]   # More stable with single samples
beta=[0.7, 0.0, 0.3]    # Include current gradient
decay_factor_f=0.7
decay_factor_b=0.7
```

**Why this works:** With single samples, relying more on long-term statistics provides better normalization estimates.

#### For Convolutional Networks

**Standard setting:**
```python
alpha=[0.5, 0.5, 0.0]   # Works well for CNNs
beta=[0.5, 0.5, 0.0]    # Equal mix
decay_factor_f=0.7
decay_factor_b=0.7
```

**For recurrent CNNs (layer-shared ResNet):**
```python
alpha=[0.7, 0.3, 0.0]   # More stable for recurrent
beta=[0.7, 0.0, 0.3]    # Include current gradient
decay_factor_f=0.7
decay_factor_b=0.7
```

#### Experimental: Using Current Input (α₃ ≠ 0)

You can experiment with including the current input directly:

```python
alpha=[0.3, 0.3, 0.4]   # Include current input
beta=[0.4, 0.2, 0.4]    # More balanced gradient mix
```

**When to use:** If you want more immediate adaptation to current batch statistics. This can be useful for non-stationary data distributions.

### Special Cases

#### Reducing to BatchNorm

If you set:
```python
alpha=[0.0, 1.0, 0.0]   # Only short-term (current batch)
beta=[0.0, 0.0, 1.0]    # Only current gradient
```

Streaming Normalization reduces to General Batch Normalization (and thus BatchNorm as a special case).

#### Ignoring Gradients Beyond Current Batch

If you set:
```python
beta=[0.0, 0.0, 1.0]    # Only current gradient
```

This ignores all gradient history and only uses the current gradient. The paper found this works reasonably well but combining with long-term gradients (β₁ > 0) generally performs better.

### Decay Factors (κ₁, κ₂, κ₃, κ₄)

Control how quickly long-term statistics adapt:

- **κ₁** (forward): How quickly forward statistics adapt
- **κ₃** (backward): How quickly backward statistics adapt
- **κ₂ = 1 - κ₁**, **κ₄ = 1 - κ₃** (automatically set)

**Recommended values:**
- **0.7** (default): Balanced adaptation - works well for most cases
- **0.9**: Slower adaptation, more stable - use for very noisy data
- **0.5**: Faster adaptation - use when data distribution changes quickly

**Note:** The paper found that κ₁ = κ₃ = 0.7 works well across different architectures.

## Training

### Standard Training Loop

Streaming Normalization works automatically with standard PyTorch training loops:

```python
import torch
import torch.nn as nn
from snorm import StreamingNorm2d

# Your model with streaming norm
model = YourModel()  # Uses StreamingNorm2d internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Standard training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # weight_update_detector is handled automatically
```

### Online Learning

Streaming Normalization naturally supports online learning (batch size = 1):

```python
# Online learning with batch size = 1
for sample, target in online_loader:  # batch_size=1
    optimizer.zero_grad()
    output = model(sample)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Recurrent Networks

Streaming Normalization is particularly effective for recurrent networks:

```python
import torch
import torch.nn as nn
from snorm import StreamingNorm1d

class NormalizedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # Apply streaming norm to hidden-to-hidden connections
        self.norm_h = StreamingNorm1d(hidden_size)
        
    def forward(self, x):
        # x: (seq_len, batch, input_size)
        h = torch.zeros(x.size(1), self.hidden_size)
        outputs = []
        for t in range(x.size(0)):
            h = self.rnn.cell(x[t], h)
            h = self.norm_h(h)  # Normalize hidden state
            outputs.append(h)
        return torch.stack(outputs)
```

## Evaluation Mode

Streaming Normalization respects `model.eval()` mode:

```python
model.eval()  # Disables tracking of new statistics
with torch.no_grad():
    output = model(x)
```

In eval mode, streaming norm uses the existing long-term running averages without updating them.

## Demo: CIFAR-10 Training

We provide a complete training demo with ResNet and MLP architectures:

### Run with Streaming Normalization

```bash
# ResNet with streaming norm
python snorm/cifar10_demo.py \
  --architecture resnet_share \
  --dataset cifar10 \
  --norm_type snorm \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.1

# MLP with streaming norm
python snorm/cifar10_demo.py \
  --architecture plain_mlp_share \
  --dataset cifar10 \
  --norm_type snorm \
  --num_layers 5 \
  --hidden_size 512 \
  --epochs 100
```

### Compare with Other Normalization Methods

```bash
# BatchNorm
python snorm/cifar10_demo.py --norm_type batchnorm --epochs 100

# LayerNorm
python snorm/cifar10_demo.py --norm_type layernorm --epochs 100

# GroupNorm
python snorm/cifar10_demo.py --norm_type groupnorm --epochs 100

# No normalization
python snorm/cifar10_demo.py --norm_type none --epochs 100
```

### Tune Hyperparameters

```bash
python snorm/cifar10_demo.py \
  --norm_type snorm \
  --decay_factor_f 0.8 \
  --decay_factor_b 0.8 \
  --snorm_alpha 0.7 0.3 0.0 \
  --snorm_beta 0.7 0.0 0.3 \
  --epochs 100
```

## Key Differences from the Paper

The original paper was implemented in **Matlab**. This Python/PyTorch implementation:

1. **Uses PyTorch autograd**: Implements streaming normalization as a custom autograd function
2. **Automatic weight update detection**: The `weight_update_detector` is handled automatically - no manual setting required
3. **Efficient storage**: Uses incremental averaging instead of storing full history (more memory efficient)
4. **L2 normalization**: Currently implements L2 normalization (standard deviation). L1 normalization can be added as an extension.

## Comparison with Other Methods

### Batch Normalization

**BatchNorm limitations:**
- ❌ Requires large mini-batches
- ❌ Doesn't work with online learning
- ❌ Doesn't work with recurrent networks (without time-specific variants)

**StreamingNorm advantages:**
- ✅ Works with any batch size (including batch size = 1)
- ✅ Works with online learning
- ✅ Works with recurrent networks out of the box

### Layer Normalization

**LayerNorm limitations:**
- ⚠️ Doesn't work well with convolutional networks
- ⚠️ Doesn't maintain long-term statistics

**StreamingNorm advantages:**
- ✅ Works well with convolutional networks
- ✅ Maintains long-term statistics for better generalization

## Biological Plausibility

As discussed in the paper, Streaming Normalization is more biologically-plausible than time-specific Batch Normalization:

1. **Single set of statistics**: Maintains one set of normalization statistics for all timesteps (like homeostatic plasticity)
2. **Online updates**: Updates statistics in a pure online fashion
3. **Neuron-wise normalization**: Can be applied neuron-wise, similar to synaptic scaling mechanisms

## Citation

If you use Streaming Normalization in your research, please cite:

```bibtex
@misc{liao2016streaming,
  title={Streaming Normalization: Towards Simpler and More Biologically-plausible Normalizations for Online and Recurrent Learning},
  author={Liao, Qianli and Kawaguchi, Kenji and Poggio, Tomaso},
  journal={CBMM Memo No. 057},
  year={2016},
  eprint={1610.06160},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Paper Abstract

> We systematically explored a spectrum of normalization algorithms related to Batch Normalization (BN) and propose a generalized formulation that simultaneously solves two major limitations of BN: (1) online learning and (2) recurrent learning. Our proposal is simpler and more biologically-plausible. Unlike previous approaches, our technique can be applied out of the box to all learning scenarios (e.g., online learning, batch learning, fully-connected, convolutional, feedforward, recurrent and mixed --- recurrent and convolutional) and compare favorably with existing approaches.

## Files

- `streaming.py`: Core streaming operation implementation
- `streaming_norm.py`: Streaming normalization layers (`StreamingNorm1d`, `StreamingNorm2d`)
- `architectures.py`: Pre-built architectures with normalization support
- `cifar10_demo.py`: Complete training demo with CIFAR-10
- `__init__.py`: Package initialization


## Acknowledgments

This work was supported by the Center for Brains, Minds and Machines (CBMM), funded by NSF STC award CCF - 1231216.

## References

- Original Paper: [arXiv:1610.06160](https://arxiv.org/abs/1610.06160)
- Batch Normalization: [Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)
- Layer Normalization: [Ba et al., 2016](https://arxiv.org/abs/1607.06450)
