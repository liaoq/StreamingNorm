"""
Unified neural network architectures.

This module contains all model architectures that are learning-algorithm agnostic.
Models can be used with any training method (backprop, ES, etc.).

All models support various normalization types including streaming normalization (snorm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import os
import sys

# Try to import snorm
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from snorm.streaming_norm import StreamingNorm1d, StreamingNorm2d
    SNORM_AVAILABLE = True
except ImportError:
    SNORM_AVAILABLE = False
    StreamingNorm1d = None
    StreamingNorm2d = None


def create_norm_layer(
    norm_type: str, 
    normalized_shape: int, 
    num_groups: int = None,
    snorm_decay_factor_f: float = 0.7,
    snorm_decay_factor_b: float = 0.7,
    snorm_alpha: List[float] = None,
    snorm_beta: List[float] = None,
    snorm_eps: float = 1e-5,
    snorm_affine: bool = True,
    is_2d: bool = False,
):
    """Create a normalization layer"""
    if norm_type is None or norm_type == "none":
        return None
    elif norm_type == "layernorm":
        if is_2d:
            # For 2D, use GroupNorm with 1 group (equivalent to LayerNorm)
            return nn.GroupNorm(1, normalized_shape)
        else:
            return nn.LayerNorm(normalized_shape)
    elif norm_type == "rmsnorm":
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        # PyTorch doesn't have native RMSNorm, so we'll create a simple version
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
            
            def forward(self, x):
                # Compute RMS: sqrt(mean(x^2))
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                # Normalize and scale
                return self.weight * (x / rms)
        return RMSNorm(normalized_shape)
    elif norm_type == "batchnorm":
        if is_2d:
            return nn.BatchNorm2d(normalized_shape)
        else:
            # For 1D data (MLP), use BatchNorm1d
            return nn.BatchNorm1d(normalized_shape)
    elif norm_type == "groupnorm":
        if num_groups is None:
            num_groups = min(32, normalized_shape)  # Default: use up to 32 groups
        return nn.GroupNorm(num_groups, normalized_shape)
    elif norm_type == "snorm":
        if not SNORM_AVAILABLE:
            raise ImportError("snorm package not available. Please install it or use a different normalization type.")
        if snorm_alpha is None:
            snorm_alpha = [0.5, 0.5, 0.0]
        if snorm_beta is None:
            snorm_beta = [0.6, 0.0, 0.4]
        if is_2d:
            return StreamingNorm2d(
                num_features=normalized_shape,
                eps=snorm_eps,
                affine=snorm_affine,
                decay_factor_f=snorm_decay_factor_f,
                decay_factor_b=snorm_decay_factor_b,
                alpha=snorm_alpha,
                beta=snorm_beta,
            )
        else:
            return StreamingNorm1d(
                num_features=normalized_shape,
                eps=snorm_eps,
                affine=snorm_affine,
                decay_factor_f=snorm_decay_factor_f,
                decay_factor_b=snorm_decay_factor_b,
                alpha=snorm_alpha,
                beta=snorm_beta,
            )
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def make_norm_layer_factory(
    norm_type: str,
    snorm_decay_factor_f: float = 0.7,
    snorm_decay_factor_b: float = 0.7,
    snorm_alpha: List[float] = None,
    snorm_beta: List[float] = None,
    snorm_eps: float = 1e-5,
    snorm_affine: bool = True,
):
    """Create a factory function for creating norm layers with snorm parameters baked in"""
    def factory(normalized_shape: int, num_groups: int = None, is_2d: bool = False):
        return create_norm_layer(
            norm_type=norm_type,
            normalized_shape=normalized_shape,
            num_groups=num_groups,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=is_2d,
        )
    return factory


class SwiGLU(nn.Module):
    """SwiGLU MLP variant (same as QWen 2.5)"""
    def __init__(
        self, 
        hidden_size: int, 
        expansion: float = 4.0, 
        norm_type: str = None,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Add normalization after gate_up_proj if specified
        self.norm = create_norm_layer(
            norm_type, 2 * intermediate_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        if self.norm is not None:
            # Apply norm to concatenated gate and up before chunking
            gate_up = torch.cat([gate, up], dim=-1)
            gate_up = self.norm(gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MultiLayerSwiGLUMLP(nn.Module):
    """Multi-layer MLP network with SwiGLU layers"""
    def __init__(
        self,
        input_size: int = 784,  # 28*28 for MNIST
        hidden_size: int = 512,
        num_layers: int = 3,
        num_classes: int = 10,
        expansion: float = 4.0,
        norm_type: str = None,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        
        # Input normalization
        self.input_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
        
        # SwiGLU layers
        self.layers = nn.ModuleList([
            SwiGLU(
                hidden_size, expansion=expansion, norm_type=norm_type,
                snorm_decay_factor_f=snorm_decay_factor_f,
                snorm_decay_factor_b=snorm_decay_factor_b,
                snorm_alpha=snorm_alpha,
                snorm_beta=snorm_beta,
                snorm_eps=snorm_eps,
                snorm_affine=snorm_affine,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization (after each SwiGLU layer)
        self.layer_norms = nn.ModuleList([
            create_norm_layer(
                norm_type, hidden_size,
                snorm_decay_factor_f=snorm_decay_factor_f,
                snorm_decay_factor_b=snorm_decay_factor_b,
                snorm_alpha=snorm_alpha,
                snorm_beta=snorm_beta,
                snorm_eps=snorm_eps,
                snorm_affine=snorm_affine,
                is_2d=False,
            ) if norm_type else None
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_classes, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        
        # SwiGLU layers with normalization
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = x + residual  # Residual connection
        
        # Output projection
        logits = self.output_proj(x)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        
        # SwiGLU layers with normalization
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = x + residual  # Residual connection
        
        return x  # [batch, hidden_size]


class MultiLayerSwiGLUMLPShare(nn.Module):
    """Multi-layer MLP network with SwiGLU layers - shared layer version (reuses same layer object)"""
    def __init__(
        self,
        input_size: int = 784,  # 28*28 for MNIST
        hidden_size: int = 512,
        num_layers: int = 3,
        num_classes: int = 10,
        expansion: float = 4.0,
        norm_type: str = None,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        
        # Input normalization
        self.input_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
        
        # Single shared SwiGLU layer (reused num_layers times)
        self.layer = SwiGLU(
            hidden_size, expansion=expansion, norm_type=norm_type,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
        )
        
        # Single shared layer normalization (reused num_layers times)
        self.layer_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        ) if norm_type else None
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_classes, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        
        # Reuse the same SwiGLU layer num_layers times
        for _ in range(self.num_layers):
            residual = x
            x = self.layer(x)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            x = x + residual  # Residual connection
        
        # Output projection
        logits = self.output_proj(x)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        
        # Reuse the same SwiGLU layer num_layers times
        for _ in range(self.num_layers):
            residual = x
            x = self.layer(x)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            x = x + residual  # Residual connection
        
        return x  # [batch, hidden_size]


class PlainMLP(nn.Module):
    """Plain MLP with standard linear layers and activation"""
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 512,
        num_layers: int = 3,
        num_classes: int = 10,
        activation: str = "relu",
        norm_type: str = None,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size, bias=True)
        
        # Input normalization
        self.input_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
        
        # Hidden layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization (after each hidden layer)
        self.layer_norms = nn.ModuleList([
            create_norm_layer(
                norm_type, hidden_size,
                snorm_decay_factor_f=snorm_decay_factor_f,
                snorm_decay_factor_b=snorm_decay_factor_b,
                snorm_alpha=snorm_alpha,
                snorm_beta=snorm_beta,
                snorm_eps=snorm_eps,
                snorm_affine=snorm_affine,
                is_2d=False,
            ) if norm_type else None
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.activation(x)
        
        # Hidden layers with normalization and residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            x_residual = x
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = self.activation(x)
            x = x + x_residual  # Residual connection
        
        # Output projection
        logits = self.output_proj(x)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.activation(x)
        
        # Hidden layers with normalization and residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            x_residual = x
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = self.activation(x)
            x = x + x_residual  # Residual connection
        
        return x  # [batch, hidden_size]


class PlainMLPShare(nn.Module):
    """Plain MLP with standard linear layers and activation - shared layer version (reuses same layer object)"""
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 512,
        num_layers: int = 3,
        num_classes: int = 10,
        activation: str = "relu",
        norm_type: str = None,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size, bias=True)
        
        # Input normalization
        self.input_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
        
        # Single shared hidden layer (reused num_layers times)
        self.layer = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Single shared layer normalization (reused num_layers times)
        self.layer_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        ) if norm_type else None
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.activation(x)
        
        # Reuse the same hidden layer num_layers times
        for _ in range(self.num_layers):
            residual = x
            x = self.layer(x)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            x = self.activation(x)
            x = x + residual  # Residual connection
        
        # Output projection
        logits = self.output_proj(x)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Flatten input
        x = x.view(x.size(0), -1)  # [batch, input_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.activation(x)
        
        # Reuse the same hidden layer num_layers times
        for _ in range(self.num_layers):
            residual = x
            x = self.layer(x)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            x = self.activation(x)
            x = x + residual  # Residual connection
        
        return x  # [batch, hidden_size]


class ResidualBlock(nn.Module):
    """Residual block with different options"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_type: str = None,
        block_type: str = "simple",  # "simple", "tiny_resnet2", "basic"
        use_residual: bool = True,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.block_type = block_type
        
        if norm_type == "batchnorm":
            norm_layer = lambda c: nn.BatchNorm2d(c)
        elif norm_type == "layernorm":
            # For 2D, we need to use LayerNorm with proper shape
            # For simplicity, we'll use GroupNorm with 1 group
            norm_layer = lambda c: nn.GroupNorm(1, c)
        elif norm_type == "groupnorm":
            norm_layer = lambda c: nn.GroupNorm(min(32, c), c)
        elif norm_type == "snorm":
            if not SNORM_AVAILABLE:
                raise ImportError("snorm package not available. Please install it or use a different normalization type.")
            # Create a closure to capture snorm parameters
            def make_snorm_layer(c):
                return StreamingNorm2d(
                    num_features=c,
                    eps=snorm_eps,
                    affine=snorm_affine,
                    decay_factor_f=snorm_decay_factor_f,
                    decay_factor_b=snorm_decay_factor_b,
                    alpha=snorm_alpha if snorm_alpha is not None else [0.5, 0.5, 0.0],
                    beta=snorm_beta if snorm_beta is not None else [0.6, 0.0, 0.4],
                )
            norm_layer = make_snorm_layer
        else:
            norm_layer = lambda c: None
        
        if block_type == "simple":
            # Simple residual block: 3x3 conv with norm and activation
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.norm = norm_layer(out_channels) if norm_layer else None
            self.activation = nn.ReLU(inplace=True)
            
            # Shortcut connection if needed
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut = None
                
        elif block_type == "tiny_resnet2":
            # Similar to tiny_resnet2.py: depthwise conv + 1x1 conv
            self.shortcut_path = nn.Sequential()
            if stride > 1:
                self.shortcut_path.add_module("dw_conv", nn.Conv2d(in_channels, in_channels, kernel_size=stride, stride=stride, groups=in_channels, bias=False))
            
            self.residual_path = nn.Sequential(
                norm_layer(in_channels) if norm_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=2 + stride, stride=stride, padding=1, groups=in_channels, bias=False),
                norm_layer(in_channels) if norm_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
            self.γ = nn.Parameter(torch.tensor(0.0))
            self.shortcut = None  # Handled in shortcut_path
            
        elif block_type == "basic":
            # Basic ResNet block: 3x3 conv -> norm -> activation -> 3x3 conv -> norm
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.norm1 = norm_layer(out_channels) if norm_layer else None
            self.activation1 = nn.ReLU(inplace=True)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm2 = norm_layer(out_channels) if norm_layer else None
            
            # Shortcut connection if needed
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    norm_layer(out_channels) if norm_layer else nn.Identity()
                )
            else:
                self.shortcut = None
                
            self.activation2 = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == "simple":
            residual = x
            out = self.conv(x)
            if self.norm is not None:
                out = self.norm(out)
            
            if self.use_residual:
                if self.shortcut is not None:
                    residual = self.shortcut(residual)
                out = out + residual
            
            out = self.activation(out)
            return out
            
        elif self.block_type == "tiny_resnet2":
            shortcut = x
            if len(self.shortcut_path) > 0:
                shortcut = self.shortcut_path(shortcut)
            
            residual = self.residual_path(x)
            
            if self.use_residual:
                out = shortcut + self.γ * residual
            else:
                out = self.γ * residual
            return out
            
        elif self.block_type == "basic":
            residual = x
            out = self.conv1(x)
            if self.norm1 is not None:
                out = self.norm1(out)
            out = self.activation1(out)
            
            out = self.conv2(out)
            if self.norm2 is not None:
                out = self.norm2(out)
            
            if self.use_residual:
                if self.shortcut is not None:
                    residual = self.shortcut(residual)
                out = out + residual
            
            out = self.activation2(out)
            return out


class ResNet(nn.Module):
    """ResNet architecture with multiple stages"""
    def __init__(
        self,
        input_size: tuple = (28, 28),  # (H, W)
        num_channels: int = 1,  # 1 for MNIST, 3 for CIFAR
        num_classes: int = 10,
        num_layers: list = [3, 5, 4, 7],  # Number of residual blocks per stage
        feat_num: list = [32, 64, 128, 256],  # Number of features per stage
        norm_type: str = None,
        block_type: str = "simple",  # "simple", "tiny_resnet2", "basic"
        use_residual: bool = True,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.feat_num = feat_num
        self.num_stages = len(num_layers)
        
        assert len(num_layers) == len(feat_num), "num_layers and feat_num must have the same length"
        
        # Initial stem layer
        self.stem = nn.Conv2d(num_channels, feat_num[0], kernel_size=3, padding=1, bias=False)
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.ModuleList()
            
            # Head conv layer for this stage (3x3, stride 2, pad 1)
            if i == 0:
                in_feat = feat_num[0]
            else:
                in_feat = feat_num[i-1]
            out_feat = feat_num[i]
            
            head_conv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, bias=False)
            stage.append(head_conv)
            
            # Residual blocks for this stage
            for _ in range(num_layers[i]):
                block = ResidualBlock(
                    in_channels=out_feat,
                    out_channels=out_feat,
                    stride=1,
                    norm_type=norm_type,
                    block_type=block_type,
                    use_residual=use_residual,
                    snorm_decay_factor_f=snorm_decay_factor_f,
                    snorm_decay_factor_b=snorm_decay_factor_b,
                    snorm_alpha=snorm_alpha,
                    snorm_beta=snorm_beta,
                    snorm_eps=snorm_eps,
                    snorm_affine=snorm_affine,
                )
                stage.append(block)
            
            self.stages.append(stage)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Output projection (optional, for compatibility with classifier interface)
        self.output_proj = nn.Linear(feat_num[-1], num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            # Head conv layer
            x = stage[0](x)
            
            # Residual blocks
            for block in stage[1:]:
                x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            # Head conv layer
            x = stage[0](x)
            
            # Residual blocks
            for block in stage[1:]:
                x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        return x


class ResNetShare(nn.Module):
    """ResNet architecture with shared residual blocks per stage"""
    def __init__(
        self,
        input_size: tuple = (28, 28),  # (H, W)
        num_channels: int = 1,  # 1 for MNIST, 3 for CIFAR
        num_classes: int = 10,
        num_layers: list = [3, 5, 4, 7],  # Number of residual blocks per stage
        feat_num: list = [32, 64, 128, 256],  # Number of features per stage
        norm_type: str = None,
        block_type: str = "simple",  # "simple", "tiny_resnet2", "basic"
        use_residual: bool = True,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.feat_num = feat_num
        self.num_stages = len(num_layers)
        
        assert len(num_layers) == len(feat_num), "num_layers and feat_num must have the same length"
        
        # Initial stem layer
        self.stem = nn.Conv2d(num_channels, feat_num[0], kernel_size=3, padding=1, bias=False)
        
        # Stages (each stage has one head conv and one shared residual block)
        self.stage_heads = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        
        for i in range(self.num_stages):
            # Head conv layer for this stage (3x3, stride 2, pad 1)
            if i == 0:
                in_feat = feat_num[0]
            else:
                in_feat = feat_num[i-1]
            out_feat = feat_num[i]
            
            head_conv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, bias=False)
            self.stage_heads.append(head_conv)
            
            # Single shared residual block for this stage (reused num_layers[i] times)
            block = ResidualBlock(
                in_channels=out_feat,
                out_channels=out_feat,
                stride=1,
                norm_type=norm_type,
                block_type=block_type,
                use_residual=use_residual,
                snorm_decay_factor_f=snorm_decay_factor_f,
                snorm_decay_factor_b=snorm_decay_factor_b,
                snorm_alpha=snorm_alpha,
                snorm_beta=snorm_beta,
                snorm_eps=snorm_eps,
                snorm_affine=snorm_affine,
            )
            self.stage_blocks.append(block)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Output projection (optional, for compatibility with classifier interface)
        self.output_proj = nn.Linear(feat_num[-1], num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        
        # Stages
        for i in range(self.num_stages):
            # Head conv layer
            x = self.stage_heads[i](x)
            
            # Reuse the same residual block num_layers[i] times
            for _ in range(self.num_layers[i]):
                x = self.stage_blocks[i](x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Stem
        x = self.stem(x)
        
        # Stages
        for i in range(self.num_stages):
            # Head conv layer
            x = self.stage_heads[i](x)
            
            # Reuse the same residual block num_layers[i] times
            for _ in range(self.num_layers[i]):
                x = self.stage_blocks[i](x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention and MLP"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_expansion: float = 4.0,
        norm_type: str = None,
        dropout: float = 0.1,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms (pre-norm architecture)
        self.norm1 = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        ) if norm_type else nn.LayerNorm(hidden_size)
        self.norm2 = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        ) if norm_type else nn.LayerNorm(hidden_size)
        
        # MLP (feed-forward network)
        mlp_hidden_size = int(hidden_size * mlp_expansion)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size, bias=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        
        # Pre-norm: Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection
        
        # Pre-norm: MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out  # Residual connection
        
        return x


class SimpleVisionTransformer(nn.Module):
    """Simple Vision Transformer (ViT) architecture"""
    def __init__(
        self,
        input_size: tuple = (28, 28),  # (H, W) for MNIST
        num_channels: int = 1,  # 1 for MNIST, 3 for CIFAR
        num_classes: int = 10,
        hidden_size: int = 512,
        num_layers: int = 3,  # Number of transformer encoder layers
        patch_size: int = 4,  # Patch size (e.g., 4 means 4x4 patches)
        num_heads: int = 8,  # Number of attention heads
        mlp_expansion: float = 4.0,  # MLP expansion factor in transformer blocks
        norm_type: str = None,
        dropout: float = 0.1,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # Calculate number of patches
        h_patches = input_size[0] // patch_size
        w_patches = input_size[1] // patch_size
        num_patches = h_patches * w_patches
        
        # Patch embedding: [batch, C, H, W] -> [batch, num_patches, hidden_size]
        patch_dim = num_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, hidden_size, bias=True)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))  # +1 for cls token
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_expansion=mlp_expansion,
                norm_type=norm_type,
                dropout=dropout,
                snorm_decay_factor_f=snorm_decay_factor_f,
                snorm_decay_factor_b=snorm_decay_factor_b,
                snorm_alpha=snorm_alpha,
                snorm_beta=snorm_beta,
                snorm_eps=snorm_eps,
                snorm_affine=snorm_affine,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm before classification head
        self.norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        ) if norm_type else nn.Identity()
        
        # Classification head
        self.head = nn.Linear(hidden_size, num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, C, H, W]
        batch_size = x.size(0)
        
        # Create patches: [batch, C, H, W] -> [batch, num_patches, patch_dim]
        patches = self._create_patches(x)  # [batch, num_patches, patch_dim]
        
        # Patch embedding: [batch, num_patches, patch_dim] -> [batch, num_patches, hidden_size]
        x = self.patch_embed(patches)  # [batch, num_patches, hidden_size]
        
        # Add class token: [batch, 1, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_size]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches + 1, hidden_size]
        
        # Add positional embeddings
        x = x + self.pos_embed  # [batch, num_patches + 1, hidden_size]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract class token (first token)
        x = self.norm(x)  # [batch, num_patches + 1, hidden_size]
        cls_token = x[:, 0]  # [batch, hidden_size]
        
        # Classification head
        logits = self.head(cls_token)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier (class token)"""
        batch_size = x.size(0)
        
        # Create patches: [batch, C, H, W] -> [batch, num_patches, patch_dim]
        patches = self._create_patches(x)  # [batch, num_patches, patch_dim]
        
        # Patch embedding: [batch, num_patches, patch_dim] -> [batch, num_patches, hidden_size]
        x = self.patch_embed(patches)  # [batch, num_patches, hidden_size]
        
        # Add class token: [batch, 1, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_size]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches + 1, hidden_size]
        
        # Add positional embeddings
        x = x + self.pos_embed  # [batch, num_patches + 1, hidden_size]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract class token (first token)
        x = self.norm(x)  # [batch, num_patches + 1, hidden_size]
        cls_token = x[:, 0]  # [batch, hidden_size]
        
        return cls_token  # [batch, hidden_size]
    
    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches"""
        # x: [batch, C, H, W]
        batch_size, C, H, W = x.shape
        patch_size = self.patch_size
        
        # Reshape to patches: [batch, C, H//patch_size, patch_size, W//patch_size, patch_size]
        # Then rearrange to [batch, num_patches, C*patch_size*patch_size]
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        x = x.view(batch_size, C, h_patches, patch_size, w_patches, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [batch, h_patches, w_patches, C, patch_size, patch_size]
        x = x.contiguous().view(batch_size, h_patches * w_patches, C * patch_size * patch_size)
        
        return x


class HybridCNNMLP(nn.Module):
    """Hybrid architecture: 2 conv layers + PlainMLP structure"""
    def __init__(
        self,
        input_size: tuple = (28, 28),  # (H, W) for MNIST
        num_channels: int = 1,  # 1 for MNIST, 3 for CIFAR
        num_classes: int = 10,
        hidden_size: int = 512,
        num_layers: int = 3,
        activation: str = "relu",
        norm_type: str = None,
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 2 convolutional layers with stride 2 and padding 1
        # Each conv reduces spatial size by half: H -> H/2
        # Channel progression: num_channels -> 32 -> 64
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Normalization layers for convolutions
        if norm_type == "batchnorm":
            self.conv1_norm = nn.BatchNorm2d(32)
            self.conv2_norm = nn.BatchNorm2d(64)
        elif norm_type == "groupnorm":
            self.conv1_norm = nn.GroupNorm(min(32, 32), 32)
            self.conv2_norm = nn.GroupNorm(min(32, 64), 64)
        elif norm_type == "snorm":
            if not SNORM_AVAILABLE:
                raise ImportError("snorm package not available. Please install it or use a different normalization type.")
            if snorm_alpha is None:
                snorm_alpha = [0.5, 0.5, 0.0]
            if snorm_beta is None:
                snorm_beta = [0.6, 0.0, 0.4]
            self.conv1_norm = StreamingNorm2d(
                num_features=32,
                eps=snorm_eps,
                affine=snorm_affine,
                decay_factor_f=snorm_decay_factor_f,
                decay_factor_b=snorm_decay_factor_b,
                alpha=snorm_alpha,
                beta=snorm_beta,
            )
            self.conv2_norm = StreamingNorm2d(
                num_features=64,
                eps=snorm_eps,
                affine=snorm_affine,
                decay_factor_f=snorm_decay_factor_f,
                decay_factor_b=snorm_decay_factor_b,
                alpha=snorm_alpha,
                beta=snorm_beta,
            )
        elif norm_type == "layernorm":
            # For 2D, use GroupNorm with 1 group (equivalent to LayerNorm)
            self.conv1_norm = nn.GroupNorm(1, 32)
            self.conv2_norm = nn.GroupNorm(1, 64)
        else:
            self.conv1_norm = None
            self.conv2_norm = None
        
        # Calculate flattened size after 2 conv layers with stride 2
        # After conv1: H/2, W/2
        # After conv2: H/4, W/4
        h_out = input_size[0] // 4
        w_out = input_size[1] // 4
        conv_output_size = 64 * h_out * w_out
        
        # Now use the same structure as PlainMLP
        # Input projection
        self.input_proj = nn.Linear(conv_output_size, hidden_size, bias=True)
        
        # Input normalization
        self.input_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
        
        # Hidden layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization (after each hidden layer)
        self.layer_norms = nn.ModuleList([
            create_norm_layer(
                norm_type, hidden_size,
                snorm_decay_factor_f=snorm_decay_factor_f,
                snorm_decay_factor_b=snorm_decay_factor_b,
                snorm_alpha=snorm_alpha,
                snorm_beta=snorm_beta,
                snorm_eps=snorm_eps,
                snorm_affine=snorm_affine,
                is_2d=False,
            ) if norm_type else None
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming init for ReLU activations"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Use Kaiming init for ReLU activations in conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Use Kaiming init for activations in linear layers
                if self.activation == F.relu:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2 convolutional layers with stride 2 (each reduces spatial size by half)
        x = self.conv1(x)  # [batch, 32, H/2, W/2]
        if self.conv1_norm is not None:
            x = self.conv1_norm(x)
        x = self.activation(x)
        
        x = self.conv2(x)  # [batch, 64, H/4, W/4]
        if self.conv2_norm is not None:
            x = self.conv2_norm(x)
        x = self.activation(x)
        
        # Flatten: [batch, 64, H/4, W/4] -> [batch, 64*H/4*W/4]
        x = x.view(x.size(0), -1)  # [batch, conv_output_size]
        
        # Now pass through PlainMLP structure
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.activation(x)
        
        # Hidden layers with normalization and residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            x_residual = x
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = self.activation(x)
            x = x + x_residual  # Residual connection
        
        # Output projection
        logits = self.output_proj(x)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # 2 convolutional layers with stride 2 (each reduces spatial size by half)
        x = self.conv1(x)  # [batch, 32, H/2, W/2]
        if self.conv1_norm is not None:
            x = self.conv1_norm(x)
        x = self.activation(x)
        
        x = self.conv2(x)  # [batch, 64, H/4, W/4]
        if self.conv2_norm is not None:
            x = self.conv2_norm(x)
        x = self.activation(x)
        
        # Flatten: [batch, 64, H/4, W/4] -> [batch, 64*H/4*W/4]
        x = x.view(x.size(0), -1)  # [batch, conv_output_size]
        
        # Now pass through PlainMLP structure
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_size]
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.activation(x)
        
        # Hidden layers with normalization and residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            x_residual = x
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = self.activation(x)
            x = x + x_residual  # Residual connection
        
        return x  # [batch, hidden_size]


class SimpleCNN(nn.Module):
    """Simple CNN for image classification: N conv layers + global avg pool + FC"""
    def __init__(
        self,
        input_size: tuple = (28, 28),  # (H, W) for MNIST
        num_channels: int = 1,  # 1 for MNIST, 3 for CIFAR
        num_classes: int = 10,
        hidden_size: int = 256,  # Reduced from 512 to 256
        num_layers: int = 2,  # Unused for CNN, kept for compatibility
        norm_type: str = None,
        use_flatten: bool = False,  # If True, flatten features instead of global avg pool
        num_conv_layers: int = 3,  # Number of conv2d layers (0, 1, 2, or 3)
        snorm_decay_factor_f: float = 0.7,
        snorm_decay_factor_b: float = 0.7,
        snorm_alpha: List[float] = None,
        snorm_beta: List[float] = None,
        snorm_eps: float = 1e-5,
        snorm_affine: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.use_flatten = use_flatten
        self.num_conv_layers = num_conv_layers
        
        # Channel progression based on number of conv layers
        # num_channels -> 32 -> 64 -> 128 -> 256 -> 512 -> ...
        # Extend channel sizes if needed (doubles after 128)
        channel_sizes = [num_channels, 32, 64, 128]
        if num_conv_layers > 3:
            # Extend channel progression: double each time after 128
            for i in range(4, num_conv_layers + 1):
                channel_sizes.append(channel_sizes[-1] * 2)
        
        # Create convolutional layers based on num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = channel_sizes[i]
            out_channels = channel_sizes[i + 1]
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
            self.conv_layers.append(conv)
            
            # Normalization layers
            if norm_type == "batchnorm":
                norm = nn.BatchNorm2d(out_channels)
            elif norm_type == "groupnorm":
                norm = nn.GroupNorm(min(32, out_channels), out_channels)
            elif norm_type == "snorm":
                if not SNORM_AVAILABLE:
                    raise ImportError("snorm package not available. Please install it or use a different normalization type.")
                if snorm_alpha is None:
                    snorm_alpha = [0.5, 0.5, 0.0]
                if snorm_beta is None:
                    snorm_beta = [0.6, 0.0, 0.4]
                norm = StreamingNorm2d(
                    num_features=out_channels,
                    eps=snorm_eps,
                    affine=snorm_affine,
                    decay_factor_f=snorm_decay_factor_f,
                    decay_factor_b=snorm_decay_factor_b,
                    alpha=snorm_alpha,
                    beta=snorm_beta,
                )
            elif norm_type == "layernorm":
                # For 2D, use GroupNorm with 1 group (equivalent to LayerNorm)
                norm = nn.GroupNorm(1, out_channels)
            else:
                norm = None
            self.conv_norms.append(norm)
        
        # Global average pooling (only used if not flattening and num_conv_layers > 0)
        if num_conv_layers > 0:
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_avg_pool = None
        
        # Calculate feature size after conv layers
        if num_conv_layers == 0:
            # No conv layers: just flatten input image
            # For MNIST: 1 * 28 * 28 = 784, for CIFAR: 3 * 32 * 32 = 3072
            if use_flatten:
                # Direct flatten: input_size[0] * input_size[1] * num_channels
                conv_output_size = num_channels * input_size[0] * input_size[1]
                self.fc = nn.Linear(conv_output_size, hidden_size, bias=True)
            else:
                # This shouldn't happen with 0 conv layers, but handle it
                conv_output_size = num_channels * input_size[0] * input_size[1]
                self.fc = nn.Linear(conv_output_size, hidden_size, bias=True)
        else:
            # Calculate flattened size after N conv layers with stride 2
            def calc_spatial_size(size, num_convs):
                for _ in range(num_convs):
                    size = (size + 2*1 - 3) // 2 + 1  # stride 2, padding 1, kernel 3
                return size
            
            h_out = calc_spatial_size(input_size[0], num_conv_layers)
            w_out = calc_spatial_size(input_size[1], num_conv_layers)
            final_channels = channel_sizes[num_conv_layers]  # 32, 64, or 128
            
            if use_flatten:
                # From flattened conv features to hidden_size
                conv_output_size = final_channels * h_out * w_out
                self.fc = nn.Linear(conv_output_size, hidden_size, bias=True)
            else:
                # From global avg pooled features to hidden_size
                self.fc = nn.Linear(final_channels, hidden_size, bias=True)
        
        self.fc_norm = create_norm_layer(
            norm_type, hidden_size,
            snorm_decay_factor_f=snorm_decay_factor_f,
            snorm_decay_factor_b=snorm_decay_factor_b,
            snorm_alpha=snorm_alpha,
            snorm_beta=snorm_beta,
            snorm_eps=snorm_eps,
            snorm_affine=snorm_affine,
            is_2d=False,
        )
        
        # Output layer
        self.output_proj = nn.Linear(hidden_size, num_classes, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming init for ReLU activations"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Use Kaiming init for ReLU activations in conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Use Kaiming init for ReLU activations in linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutional layers
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = F.relu(x)
        
        # Process features: either global avg pool or flatten
        if self.num_conv_layers == 0:
            # No conv layers: just flatten input image
            x = x.view(x.size(0), -1)  # [batch, num_channels * H * W]
        elif self.use_flatten:
            # Flatten: [batch, channels, H, W] -> [batch, channels * H * W]
            x = x.view(x.size(0), -1)
        else:
            # Global average pooling: [batch, channels, H, W] -> [batch, channels, 1, 1]
            x = self.global_avg_pool(x)  # [batch, channels, 1, 1]
            # Flatten: [batch, channels, 1, 1] -> [batch, channels]
            x = x.view(x.size(0), -1)  # [batch, channels]
        
        # Single FC layer
        x = self.fc(x)  # [batch, hidden_size]
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = F.relu(x)
        
        # Output projection
        logits = self.output_proj(x)  # [batch, num_classes]
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        # Apply convolutional layers
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = F.relu(x)
        
        # Process features: either global avg pool or flatten
        if self.num_conv_layers == 0:
            # No conv layers: just flatten input image
            x = x.view(x.size(0), -1)  # [batch, num_channels * H * W]
        elif self.use_flatten:
            # Flatten: [batch, channels, H, W] -> [batch, channels * H * W]
            x = x.view(x.size(0), -1)
        else:
            # Global average pooling: [batch, channels, H, W] -> [batch, channels, 1, 1]
            x = self.global_avg_pool(x)  # [batch, channels, 1, 1]
            # Flatten: [batch, channels, 1, 1] -> [batch, channels]
            x = x.view(x.size(0), -1)  # [batch, channels]
        
        # Single FC layer
        x = self.fc(x)  # [batch, hidden_size]
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = F.relu(x)
        
        return x  # [batch, hidden_size]

