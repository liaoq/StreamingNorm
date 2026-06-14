"""
Streaming operation for deep networks.

This module implements a custom PyTorch autograd operation that maintains
short-term and long-term running averages of data and gradients.
"""

import torch
import torch.nn as nn
from typing import Optional
from torch.autograd import Function


class ShortTermHistory:
    """
    Efficient storage for short-term history using running average and counter.
    
    Instead of maintaining a list of tensors, this class maintains:
    - average: Running average of all tensors seen
    - counter: Number of tensors seen
    """
    
    def __init__(self):
        """Initialize short-term history with empty state."""
        self.average: Optional[torch.Tensor] = None
        self.counter: int = 0
    
    def push(self, tensor: torch.Tensor):
        """
        Push a new tensor to the history and update the running average.
        
        Args:
            tensor: Tensor to add (will be detached)
        """
        tensor_detached = tensor.detach()
        
        if self.counter == 0:
            # First tensor: average is simply the tensor
            self.average = tensor_detached
            self.counter = 1
        else:
            # Update running average: new_average = (old_average * counter + new_tensor) / (counter + 1)
            self.average = (self.average * self.counter + tensor_detached) / (self.counter + 1)
            self.counter += 1
    
    def clear(self):
        """Clear the history by resetting average to None and counter to 0."""
        self.average = None
        self.counter = 0
    
    def get_average(self) -> Optional[torch.Tensor]:
        """
        Get the current average.
        
        Returns:
            Average tensor if counter > 0, None otherwise
        """
        return self.average
    
    def is_empty(self) -> bool:
        """Check if history is empty."""
        return self.counter == 0 or self.average is None


class StreamingFunction(Function):
    """
    Custom autograd function for streaming operation.
    
    This function maintains running averages of data (forward) and gradients (backward).
    """
    
    @staticmethod
    def forward(ctx, input_data, weight_update_detector, streaming_state):
        """
        Forward pass of streaming operation.
        
        Args:
            input_data: Input tensor to be streamed
            weight_update_detector: Trainable parameter that detects weight updates
            streaming_state: StreamingState object containing state variables
        
        Returns:
            Streamed output (detached running average)
        """
        ctx.streaming_state = streaming_state
        ctx.weight_update_detector = weight_update_detector
        
        # if in training mode
        if streaming_state.training:
            # Check if weight update occurred (weight_update_detector is nonzero)
            if weight_update_detector.item() != 0:
                # Update long-term running averages
                streaming_state.update_running_averages()
                # Clear short-term histories
                streaming_state.clear_histories()
                # Note: weight_update_detector will be reset to 0 by the optimizer
                # after it processes the gradient we provide in backward
            streaming_state.push_forward(input_data.detach())
        
        # Return weighted combination (detached)
        output = streaming_state.get_forward_output(input_data.detach())
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of streaming operation.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            Gradient for input_data, gradient for weight_update_detector, None for streaming_state
        """
        streaming_state = ctx.streaming_state
        weight_update_detector = ctx.weight_update_detector
        
        # Push gradient to short-term history if in training mode
        if streaming_state.training:
            streaming_state.push_backward(grad_output.detach())
        
        # Get weighted combination gradient
        grad_input = streaming_state.get_backward_output(grad_output.detach(), grad_output_shape=grad_output.shape)
        
        # Set constant gradient for weight_update_detector (redundancy for safety)
        grad_weight_update_detector = torch.ones_like(weight_update_detector) * streaming_state.grad_constant
        
        return grad_input, grad_weight_update_detector, None


class Streaming(nn.Module):
    """
    Streaming operation module.
    
    This module maintains short-term and long-term running averages of data and gradients.
    It uses a weight_update_detector to track when network weights are updated.
    
    Attributes:
        short_term_history_f: ShortTermHistory for forward data
        long_term_running_avg_f: Long-term running average of forward data
        short_term_history_b: ShortTermHistory for backward gradients
        long_term_running_avg_b: Long-term running average of backward gradients
        weight_update_detector: Trainable parameter that detects weight updates
        decay_factor_f: Decay factor for forward running average
        decay_factor_b: Decay factor for backward running average
        grad_constant: Constant gradient value for weight_update_detector
    """
    
    def __init__(
        self,
        decay_factor_f: float = 0.7,
        decay_factor_b: float = 0.7,
        grad_constant: float = 1.0,
        alpha: list = [0.5, 0.5, 0.0],
        beta: list = [0.6, 0.0, 0.4],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize streaming operation.
        
        Args:
            decay_factor_f: Decay factor for forward running average (default: 0.7)
            decay_factor_b: Decay factor for backward running average (default: 0.7)
            grad_constant: Constant gradient value for weight_update_detector (default: 1.0)
            alpha: Weight coefficients for forward output [long_term, short_term, current] (default: [0.5, 0.5, 0.0])
            beta: Weight coefficients for backward output [long_term, short_term, current] (default: [0.6, 0.0, 0.4])
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.decay_factor_f = decay_factor_f
        self.decay_factor_b = decay_factor_b
        self.grad_constant = grad_constant
        self.alpha = alpha  # [long_term, short_term, current]
        self.beta = beta    # [long_term, short_term, current]
        
        # Initialize state variables using efficient ShortTermHistory
        self.short_term_history_f = ShortTermHistory()
        self.long_term_running_avg_f: Optional[torch.Tensor] = None
        self.short_term_history_b = ShortTermHistory()
        self.long_term_running_avg_b: Optional[torch.Tensor] = None
        
        # Weight update detector (trainable parameter)
        # Initialize to 0. After optimizer.step() with grad=1.0, it becomes negative (e.g., -0.1 with lr=0.1)
        # This negative value (nonzero) will trigger the update check in the next forward pass
        # We then reset it to 0 after processing, and the cycle repeats
        self.weight_update_detector = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        
        # Track training mode
        self.training = True
    
    def push_forward(self, data: torch.Tensor):
        """Push data to short-term forward history."""
        self.short_term_history_f.push(data)
    
    def push_backward(self, grad: torch.Tensor):
        """Push gradient to short-term backward history."""
        self.short_term_history_b.push(grad)
    
    def clear_histories(self):
        """Clear short-term histories."""
        self.short_term_history_f.clear()
        self.short_term_history_b.clear()
    
    def update_running_averages(self):
        """Update long-term running averages from short-term histories."""
        # Update forward running average
        mean_f = self.short_term_history_f.get_average()
        if mean_f is not None:
            if self.long_term_running_avg_f is None:
                self.long_term_running_avg_f = mean_f
            else:
                self.long_term_running_avg_f = (
                    self.long_term_running_avg_f * self.decay_factor_f +
                    mean_f * (1 - self.decay_factor_f)
                )
        
        # Update backward running average
        mean_b = self.short_term_history_b.get_average()
        if mean_b is not None:
            if self.long_term_running_avg_b is None:
                self.long_term_running_avg_b = mean_b
            else:
                self.long_term_running_avg_b = (
                    self.long_term_running_avg_b * self.decay_factor_b +
                    mean_b * (1 - self.decay_factor_b)
                )
    
    def get_forward_output(self, current_f: torch.Tensor) -> torch.Tensor:
        """
        Get forward output as weighted combination: alpha[0]*long_term + alpha[1]*short_term + alpha[2]*current
        
        Args:
            current_f: Current forward data
        
        Returns:
            Weighted combination output (detached)
        """
        # Get short_term_f (directly from the maintained average)
        short_term_f = self.short_term_history_f.get_average()
        if short_term_f is None:
            # In eval mode, if short-term history is empty, use current data as fallback
            if not self.training:
                import warnings
                warnings.warn(
                    "Short-term history is empty in eval mode. Using current data for short_term_f, "
                    "which may be seriously wrong. This typically happens when forward_features is called "
                    "with dummy input before any training.",
                    UserWarning,
                    stacklevel=2
                )
                short_term_f = current_f.detach()
            else:
                # In training mode, this is an error
                raise ValueError("Short-term history is empty")
        short_term_f = short_term_f.detach()
        
        # Get long_term_f
        if self.long_term_running_avg_f is not None:
            long_term_f = self.long_term_running_avg_f.detach()
        else:
            # before first weight update, long_term_running_avg_f is not available, so use short_term_f
            long_term_f = short_term_f
        
        # Compute weighted combination
        output = torch.zeros_like(current_f)

        if self.alpha[0] != 0:
            output = output + self.alpha[0] * long_term_f
        if self.alpha[1] != 0:
            output = output + self.alpha[1] * short_term_f
        if self.alpha[2] != 0:
            output = output + self.alpha[2] * current_f.detach()
        
        return output
    
    def get_backward_output(self, current_b: torch.Tensor, grad_output_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """
        Get backward output as weighted combination: beta[0]*long_term + beta[1]*short_term + beta[2]*current
        
        Args:
            current_b: Current backward gradient
            grad_output_shape: Shape of grad_output (for fallback)
        
        Returns:
            Weighted combination output (detached)
        """
        # Get short_term_b (directly from the maintained average)
        short_term_b = self.short_term_history_b.get_average()
        if short_term_b is None:
            # In eval mode, if short-term history is empty, use current data as fallback
            if not self.training:
                import warnings
                warnings.warn(
                    "Short-term backward history is empty in eval mode. Using current data for short_term_b, "
                    "which may be seriously wrong.",
                    UserWarning,
                    stacklevel=2
                )
                short_term_b = current_b.detach()
            else:
                # In training mode, this is an error
                raise ValueError("Short-term history is empty")
        short_term_b = short_term_b.detach()
        
        # Get long_term_b
        if self.long_term_running_avg_b is not None:
            long_term_b = self.long_term_running_avg_b.detach()
        else:
            # before first weight update, long_term_running_avg_b is not available, so use short_term_b
            long_term_b = short_term_b

        # Compute weighted combination
        output = torch.zeros_like(current_b)
        
        if self.beta[0] != 0:
            output = output + self.beta[0] * long_term_b
        if self.beta[1] != 0:
            output = output + self.beta[1] * short_term_b
        if self.beta[2] != 0:
            output = output + self.beta[2] * current_b.detach()
        return output
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_data: Input tensor
        
        Returns:
            Streamed output (detached running average)
        """
        result = StreamingFunction.apply(input_data, self.weight_update_detector, self)
        
        # # Reset weight_update_detector to 0 after checking (if it was nonzero)
        # # We do this after the forward pass to avoid modifying it during the graph
        # if self.weight_update_detector.item() != 0:
        #     with torch.no_grad():
        #         self.weight_update_detector.zero_()
        
        return result
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

