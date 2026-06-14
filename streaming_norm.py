"""
Streaming Normalization layers.

This module implements normalization layers that use streaming operations
to maintain running averages of mean and standard deviation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .streaming import Streaming


class StreamingNorm1d(nn.Module):
    """
    1D Streaming Normalization layer.
    
    This layer normalizes input data using streaming mean and standard deviation.
    Similar to BatchNorm1d but uses streaming operation for smoothing.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        decay_factor_f: float = 0.7,
        decay_factor_b: float = 0.7,
        alpha: list = [0.5, 0.5, 0.0],
        beta: list = [0.6, 0.0, 0.4],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize StreamingNorm1d.
        
        Args:
            num_features: Number of features/channels
            eps: Small value to avoid division by zero (default: 1e-5)
            affine: If True, apply learnable affine transformation (default: True)
            decay_factor_f: Decay factor for forward streaming (default: 0.7)
            decay_factor_b: Decay factor for backward streaming (default: 0.7)
            alpha: Weight coefficients for forward output [long_term, short_term, current] (default: [0.5, 0.5, 0.0])
            beta: Weight coefficients for backward output [long_term, short_term, current] (default: [0.6, 0.0, 0.4])
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # Streaming operations for mean and std
        self.streaming_mean = Streaming(
            decay_factor_f=decay_factor_f,
            decay_factor_b=decay_factor_b,
            alpha=alpha,
            beta=beta,
            device=device,
            dtype=dtype,
        )
        self.streaming_std = Streaming(
            decay_factor_f=decay_factor_f,
            decay_factor_b=decay_factor_b,
            alpha=alpha,
            beta=beta,
            device=device,
            dtype=dtype,
        )
        
        # Affine transformation parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, C) or (N, C, L)
        
        Returns:
            Normalized tensor
        """
        # Compute mean and std along appropriate dimensions
        if x.dim() == 2:
            # (N, C) - normalize along batch dimension
            data_mean = x.mean(dim=0, keepdim=True)  # (1, C)
            data_std = x.std(dim=0, keepdim=True) + self.eps  # (1, C)
        elif x.dim() == 3:
            # (N, C, L) - normalize along batch and spatial dimensions
            data_mean = x.mean(dim=(0, 2), keepdim=True)  # (1, C, 1)
            data_std = x.std(dim=(0, 2), keepdim=True) + self.eps  # (1, C, 1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Stream the mean and std
        #import pdb; pdb.set_trace()

        data_mean_stream = self.streaming_mean(data_mean)
        data_std_stream = self.streaming_std(data_std)
        
        # Normalize (add epsilon to std to prevent division by zero)
        x_norm = (x - data_mean_stream) / (data_std_stream + self.eps)
        
        # Apply affine transformation if enabled
        if self.affine:
            if x.dim() == 2:
                x_norm = x_norm * self.weight + self.bias
            else:
                x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        return x_norm


class StreamingNorm2d(nn.Module):
    """
    2D Streaming Normalization layer.
    
    This layer normalizes input data using streaming mean and standard deviation.
    Similar to BatchNorm2d but uses streaming operation for smoothing.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        decay_factor_f: float = 0.7,
        decay_factor_b: float = 0.7,
        alpha: list = [0.5, 0.5, 0.0],
        beta: list = [0.6, 0.0, 0.4],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize StreamingNorm2d.
        
        Args:
            num_features: Number of features/channels
            eps: Small value to avoid division by zero (default: 1e-5)
            affine: If True, apply learnable affine transformation (default: True)
            decay_factor_f: Decay factor for forward streaming (default: 0.7)
            decay_factor_b: Decay factor for backward streaming (default: 0.7)
            alpha: Weight coefficients for forward output [long_term, short_term, current] (default: [0.5, 0.5, 0.0])
            beta: Weight coefficients for backward output [long_term, short_term, current] (default: [0.6, 0.0, 0.4])
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # Streaming operations for mean and std
        self.streaming_mean = Streaming(
            decay_factor_f=decay_factor_f,
            decay_factor_b=decay_factor_b,
            alpha=alpha,
            beta=beta,
            device=device,
            dtype=dtype,
        )
        self.streaming_std = Streaming(
            decay_factor_f=decay_factor_f,
            decay_factor_b=decay_factor_b,
            alpha=alpha,
            beta=beta,
            device=device,
            dtype=dtype,
        )
        
        # Affine transformation parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
        
        Returns:
            Normalized tensor
        """
        # Compute mean and std along batch and spatial dimensions
        # Normalize along (N, H, W) dimensions, keeping C
        data_mean = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        data_std = x.std(dim=(0, 2, 3), keepdim=True) + self.eps  # (1, C, 1, 1)
        
        # Stream the mean and std
        data_mean_stream = self.streaming_mean(data_mean)
        data_std_stream = self.streaming_std(data_std)
        
        # Normalize (add epsilon to std to prevent division by zero)
        x_norm = (x - data_mean_stream) / (data_std_stream + self.eps)
        
        # Apply affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x_norm

