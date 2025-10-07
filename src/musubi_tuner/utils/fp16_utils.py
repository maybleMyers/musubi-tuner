"""
FP16 Training Utilities for Stable Mixed Precision Training

This module provides utilities for training neural networks in fp16 precision
while maintaining numerical stability. Inspired by llm.c's approach to fp16 training.

Key components:
- Dynamic loss scaling to prevent gradient underflow
- Master weight management for fp32 optimizer states
- Numerical stability utilities
- Gradient overflow detection and recovery
- Stochastic rounding for reduced bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DynamicLossScaler:
    """
    Dynamic loss scaling for fp16 training to prevent gradient underflow.
    
    Similar to llm.c's approach but adapted for PyTorch. Automatically adjusts
    the loss scale based on gradient overflow detection.
    """
    
    def __init__(
        self,
        init_scale: float = 2**15,  # Conservative starting scale
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
        enabled: bool = True
    ):
        """
        Initialize the dynamic loss scaler.
        
        Args:
            init_scale: Initial loss scale value
            scale_factor: Factor to adjust scale by
            scale_window: Number of iterations before increasing scale
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
            enabled: Whether scaling is enabled
        """
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.enabled = enabled
        
        self._iter_since_update = 0
        self._overflow_count = 0
        self._underflow_count = 0
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss for backward pass."""
        if not self.enabled:
            return loss
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before optimizer step."""
        if not self.enabled:
            return
            
        inv_scale = 1.0 / self.scale
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
    
    def check_gradients(self, parameters) -> Tuple[bool, bool]:
        """
        Check gradients for overflow (NaN/Inf) and underflow.
        
        Returns:
            (has_overflow, has_underflow)
        """
        has_overflow = False
        has_underflow = False
        
        for param in parameters:
            if param.grad is not None:
                grad_data = param.grad.data
                
                # Check for NaN or Inf
                if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                    has_overflow = True
                
                # Check for underflow (gradients too small)
                grad_norm = grad_data.norm()
                if grad_norm > 0 and grad_norm < 1e-8:
                    has_underflow = True
                    
        return has_overflow, has_underflow
    
    def update_scale(self, has_overflow: bool, has_underflow: bool = False) -> None:
        """Update the loss scale based on gradient status."""
        if not self.enabled:
            return
            
        if has_overflow:
            # Decrease scale on overflow
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            self._iter_since_update = 0
            self._overflow_count += 1
            logger.debug(f"Gradient overflow detected. Scale decreased to {self.scale:.1f}")
            
        elif has_underflow and self.scale < self.max_scale / self.scale_factor:
            # Increase scale on underflow
            self.scale = min(self.scale * self.scale_factor, self.max_scale)
            self._iter_since_update = 0
            self._underflow_count += 1
            logger.debug(f"Gradient underflow detected. Scale increased to {self.scale:.1f}")
            
        else:
            # Gradually increase scale if no issues
            self._iter_since_update += 1
            if self._iter_since_update >= self.scale_window:
                self.scale = min(self.scale * self.scale_factor, self.max_scale)
                self._iter_since_update = 0
                logger.debug(f"Scale increased to {self.scale:.1f} after {self.scale_window} stable iterations")
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'scale': self.scale,
            '_iter_since_update': self._iter_since_update,
            '_overflow_count': self._overflow_count,
            '_underflow_count': self._underflow_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.scale = state_dict['scale']
        self._iter_since_update = state_dict['_iter_since_update']
        self._overflow_count = state_dict.get('_overflow_count', 0)
        self._underflow_count = state_dict.get('_underflow_count', 0)


class FP16MasterWeights:
    """
    Maintains fp32 master copies of fp16 model weights for stable optimization.
    
    Following llm.c's pattern of keeping master weights in fp32 for AdamW updates,
    then casting back to fp16 with optional stochastic rounding.
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_stochastic_rounding: bool = True,
        rng_seed: Optional[int] = None
    ):
        """
        Initialize master weight management.
        
        Args:
            model: The model with fp16 weights
            use_stochastic_rounding: Whether to use stochastic rounding
            rng_seed: Seed for stochastic rounding RNG
        """
        self.use_stochastic_rounding = use_stochastic_rounding
        self.rng = np.random.RandomState(rng_seed) if rng_seed else np.random.RandomState()
        
        self.master_weights = {}
        self.model_to_master = {}
        
        # Create fp32 copies of all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                master = param.detach().float().clone()
                master.requires_grad = True
                self.master_weights[name] = master
                self.model_to_master[param] = master
    
    def zero_grad(self) -> None:
        """Zero gradients on master weights."""
        for master in self.master_weights.values():
            if master.grad is not None:
                master.grad.zero_()
    
    def copy_gradients_to_master(self, model: nn.Module) -> None:
        """Copy gradients from fp16 model to fp32 master weights."""
        for name, param in model.named_parameters():
            if name in self.master_weights and param.grad is not None:
                master = self.master_weights[name]
                if master.grad is None:
                    master.grad = param.grad.float().clone()
                else:
                    master.grad.data.copy_(param.grad.float())
    
    def update_model_weights(self, model: nn.Module) -> None:
        """
        Update fp16 model weights from fp32 master weights.
        Uses stochastic rounding to reduce bias if enabled.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.master_weights:
                    master = self.master_weights[name]
                    if self.use_stochastic_rounding:
                        param.data = self._stochastic_round(master.data, param.dtype)
                    else:
                        param.data = master.data.to(param.dtype)
    
    def _stochastic_round(self, tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """
        Stochastic rounding for reduced bias when converting from fp32 to fp16.
        
        Following llm.c's approach: adds uniform noise before rounding to
        statistically preserve more information.
        """
        if target_dtype == torch.float16:
            # Add uniform noise in [-0.5, 0.5] before rounding
            noise = torch.rand_like(tensor) - 0.5
            # Scale noise based on the magnitude of values
            eps = torch.finfo(torch.float16).eps
            noise = noise * eps * tensor.abs()
            return (tensor + noise).to(target_dtype)
        return tensor.to(target_dtype)
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get master weights for checkpointing."""
        return {name: weight.clone() for name, weight in self.master_weights.items()}
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load master weights from checkpoint."""
        for name, weight in state_dict.items():
            if name in self.master_weights:
                self.master_weights[name].data.copy_(weight)


# Numerical Stability Utilities

def fp16_clamp(x: torch.Tensor, eps: float = 1000) -> torch.Tensor:
    """
    Clamp tensor values to prevent overflow in fp16.
    
    Similar to llm.c's clamping strategy but adapted for PyTorch.
    """
    if x.dtype == torch.float16:
        # Get fp16 limits
        max_val = torch.finfo(torch.float16).max
        # Leave some headroom to prevent immediate overflow
        clamp_max = max_val - eps
        clamp_min = -clamp_max
        return torch.clamp(x, min=clamp_min, max=clamp_max)
    return x


def stable_softmax_fp16(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax for fp16.
    
    Subtracts max before exp to prevent overflow, similar to llm.c's approach.
    """
    # Cast to fp32 for stability
    logits_fp32 = logits.float()
    
    # Subtract max for numerical stability
    logits_max = logits_fp32.max(dim=dim, keepdim=True)[0]
    logits_shifted = logits_fp32 - logits_max
    
    # Compute softmax in fp32
    exp_logits = torch.exp(logits_shifted)
    softmax = exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + 1e-8)
    
    # Cast back to original dtype
    return softmax.to(logits.dtype)


def compute_loss_fp16_safe(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: str = "mse",
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute loss with fp16 safety measures.
    
    Always computes loss in fp32 for stability, following llm.c's pattern.
    """
    # Cast to fp32 for stable loss computation
    pred_fp32 = predictions.float()
    target_fp32 = targets.float()
    
    if loss_fn == "mse":
        loss = F.mse_loss(pred_fp32, target_fp32, reduction=reduction)
    elif loss_fn == "l1":
        loss = F.l1_loss(pred_fp32, target_fp32, reduction=reduction)
    elif loss_fn == "smooth_l1":
        loss = F.smooth_l1_loss(pred_fp32, target_fp32, reduction=reduction)
    elif loss_fn == "cross_entropy":
        loss = F.cross_entropy(pred_fp32, target_fp32, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    
    # Check for NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(f"NaN/Inf detected in {loss_fn} loss, returning zero loss")
        return torch.tensor(0.0, device=loss.device, dtype=torch.float32)
    
    return loss


def check_model_health(model: nn.Module) -> Dict[str, Any]:
    """
    Check model weights and gradients for NaN/Inf and other issues.
    
    Returns a health report dictionary.
    """
    health = {
        'has_nan_weights': False,
        'has_inf_weights': False,
        'has_nan_grads': False,
        'has_inf_grads': False,
        'has_zero_grads': False,
        'max_weight_norm': 0.0,
        'max_grad_norm': 0.0,
        'num_nan_weights': 0,
        'num_inf_weights': 0,
        'num_nan_grads': 0,
        'num_inf_grads': 0,
    }
    
    for name, param in model.named_parameters():
        # Check weights
        if torch.isnan(param).any():
            health['has_nan_weights'] = True
            health['num_nan_weights'] += torch.isnan(param).sum().item()
            logger.warning(f"NaN weights detected in {name}")
            
        if torch.isinf(param).any():
            health['has_inf_weights'] = True
            health['num_inf_weights'] += torch.isinf(param).sum().item()
            logger.warning(f"Inf weights detected in {name}")
        
        weight_norm = param.data.norm().item()
        health['max_weight_norm'] = max(health['max_weight_norm'], weight_norm)
        
        # Check gradients
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                health['has_nan_grads'] = True
                health['num_nan_grads'] += torch.isnan(param.grad).sum().item()
                logger.warning(f"NaN gradients detected in {name}")
                
            if torch.isinf(param.grad).any():
                health['has_inf_grads'] = True
                health['num_inf_grads'] += torch.isinf(param.grad).sum().item()
                logger.warning(f"Inf gradients detected in {name}")
            
            if param.grad.norm().item() == 0:
                health['has_zero_grads'] = True
            
            grad_norm = param.grad.norm().item()
            health['max_grad_norm'] = max(health['max_grad_norm'], grad_norm)
    
    health['is_healthy'] = not any([
        health['has_nan_weights'],
        health['has_inf_weights'],
        health['has_nan_grads'],
        health['has_inf_grads']
    ])
    
    return health


@contextmanager
def fp16_autocast_context(
    device_type: str = "cuda",
    enabled: bool = True,
    cache_enabled: bool = True
):
    """
    Context manager for fp16 autocast with safety checks.
    
    Provides a safe context for mixed precision operations.
    """
    if enabled and device_type == "cuda":
        # Use PyTorch's native autocast for fp16
        with torch.cuda.amp.autocast(
            enabled=True,
            dtype=torch.float16,
            cache_enabled=cache_enabled
        ):
            yield
    else:
        # No autocast
        yield


class FP16SafeOptimizer:
    """
    Wrapper for optimizers to work with fp16 training and master weights.
    
    Implements the optimization strategy from llm.c with fp32 master weights
    and fp16 model weights.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        master_weights: FP16MasterWeights,
        loss_scaler: Optional[DynamicLossScaler] = None,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize the fp16-safe optimizer wrapper.
        
        Args:
            optimizer: Base optimizer (should operate on master weights)
            master_weights: Master weight manager
            loss_scaler: Optional loss scaler
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.optimizer = optimizer
        self.master_weights = master_weights
        self.loss_scaler = loss_scaler
        self.max_grad_norm = max_grad_norm
        
        # Replace optimizer's parameters with master weights
        self._replace_optimizer_params()
    
    def _replace_optimizer_params(self):
        """Replace optimizer's parameters with master weights."""
        master_params = list(self.master_weights.master_weights.values())
        
        # Clear existing param groups
        self.optimizer.param_groups.clear()
        
        # Add master weights as parameters
        self.optimizer.add_param_group({'params': master_params})
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
        self.master_weights.zero_grad()
    
    def step(self, model: nn.Module) -> bool:
        """
        Perform optimization step with fp16 safety.
        
        Args:
            model: The fp16 model
            
        Returns:
            True if step was successful, False if skipped due to overflow
        """
        # Copy gradients to master weights
        self.master_weights.copy_gradients_to_master(model)
        
        # Check for gradient overflow
        has_overflow = False
        has_underflow = False
        
        if self.loss_scaler:
            # Unscale gradients
            self.loss_scaler.unscale_gradients(self.optimizer)
            
            # Check gradient health
            master_params = self.master_weights.master_weights.values()
            has_overflow, has_underflow = self.loss_scaler.check_gradients(master_params)
        
        if not has_overflow:
            # Clip gradients (on master weights)
            if self.max_grad_norm > 0:
                master_params = list(self.master_weights.master_weights.values())
                torch.nn.utils.clip_grad_norm_(master_params, self.max_grad_norm)
            
            # Optimizer step on master weights
            self.optimizer.step()
            
            # Update model weights from master
            self.master_weights.update_model_weights(model)
        
        # Update loss scale
        if self.loss_scaler:
            self.loss_scaler.update_scale(has_overflow, has_underflow)
        
        return not has_overflow
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'master_weights': self.master_weights.state_dict(),
        }
        if self.loss_scaler:
            state['loss_scaler'] = self.loss_scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.master_weights.load_state_dict(state_dict['master_weights'])
        if self.loss_scaler and 'loss_scaler' in state_dict:
            self.loss_scaler.load_state_dict(state_dict['loss_scaler'])


def create_fp16_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    use_master_weights: bool = True,
    use_loss_scaling: bool = True,
    max_grad_norm: float = 1.0,
    **kwargs
) -> Union[FP16SafeOptimizer, torch.optim.Optimizer]:
    """
    Create an optimizer configured for fp16 training.
    
    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for Adam
        eps: Epsilon for Adam
        use_master_weights: Whether to use fp32 master weights
        use_loss_scaling: Whether to use dynamic loss scaling
        max_grad_norm: Maximum gradient norm for clipping
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer (wrapped if using master weights)
    """
    if use_master_weights:
        # Create master weight manager
        master_weights = FP16MasterWeights(model)
        
        # Create loss scaler if requested
        loss_scaler = DynamicLossScaler() if use_loss_scaling else None
        
        # Create base optimizer on master weights
        master_params = list(master_weights.master_weights.values())
        
        if optimizer_type.lower() == "adamw":
            base_optimizer = torch.optim.AdamW(
                master_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == "adam":
            base_optimizer = torch.optim.Adam(
                master_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            base_optimizer = torch.optim.SGD(
                master_params,
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Wrap in FP16SafeOptimizer
        return FP16SafeOptimizer(
            base_optimizer,
            master_weights,
            loss_scaler,
            max_grad_norm
        )
    else:
        # Standard optimizer without master weights
        if optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Gradient accumulation utilities

class FP16GradientAccumulator:
    """
    Manages gradient accumulation with proper scaling for fp16 training.
    """
    
    def __init__(
        self,
        accumulation_steps: int,
        loss_scaler: Optional[DynamicLossScaler] = None
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of accumulation steps
            loss_scaler: Optional loss scaler for fp16
        """
        self.accumulation_steps = accumulation_steps
        self.loss_scaler = loss_scaler
        self.current_step = 0
    
    def accumulate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for accumulation and fp16 training.
        
        Args:
            loss: The loss to accumulate
            
        Returns:
            Scaled loss for backward pass
        """
        # Scale by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Apply fp16 loss scaling if available
        if self.loss_scaler:
            scaled_loss = self.loss_scaler.scale_loss(scaled_loss)
        
        return scaled_loss
    
    def should_update(self) -> bool:
        """Check if we should perform optimizer update."""
        self.current_step += 1
        should_update = (self.current_step % self.accumulation_steps) == 0
        if should_update:
            self.current_step = 0
        return should_update
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0


# Monitoring utilities

class FP16TrainingMonitor:
    """
    Monitors fp16 training health and provides diagnostics.
    """
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize training monitor.
        
        Args:
            log_interval: How often to log statistics
        """
        self.log_interval = log_interval
        self.step = 0
        self.stats = {
            'overflow_count': 0,
            'underflow_count': 0,
            'nan_loss_count': 0,
            'inf_loss_count': 0,
            'max_grad_norm': 0.0,
            'min_grad_norm': float('inf'),
            'loss_scale_history': [],
        }
    
    def log_step(
        self,
        loss: torch.Tensor,
        grad_norm: float,
        loss_scale: float,
        has_overflow: bool,
        has_underflow: bool
    ):
        """Log statistics for current step."""
        self.step += 1
        
        # Update counters
        if has_overflow:
            self.stats['overflow_count'] += 1
        if has_underflow:
            self.stats['underflow_count'] += 1
        if torch.isnan(loss):
            self.stats['nan_loss_count'] += 1
        if torch.isinf(loss):
            self.stats['inf_loss_count'] += 1
        
        # Update gradient norms
        self.stats['max_grad_norm'] = max(self.stats['max_grad_norm'], grad_norm)
        self.stats['min_grad_norm'] = min(self.stats['min_grad_norm'], grad_norm)
        
        # Track loss scale
        self.stats['loss_scale_history'].append(loss_scale)
        if len(self.stats['loss_scale_history']) > 100:
            self.stats['loss_scale_history'].pop(0)
        
        # Log periodically
        if self.step % self.log_interval == 0:
            self.print_summary()
    
    def print_summary(self):
        """Print training health summary."""
        logger.info(
            f"FP16 Training Health - Step {self.step}: "
            f"Overflows: {self.stats['overflow_count']}, "
            f"Underflows: {self.stats['underflow_count']}, "
            f"NaN losses: {self.stats['nan_loss_count']}, "
            f"Inf losses: {self.stats['inf_loss_count']}, "
            f"Grad norm range: [{self.stats['min_grad_norm']:.2e}, {self.stats['max_grad_norm']:.2e}], "
            f"Current loss scale: {self.stats['loss_scale_history'][-1] if self.stats['loss_scale_history'] else 0:.1f}"
        )


# Export main components
__all__ = [
    'DynamicLossScaler',
    'FP16MasterWeights',
    'FP16SafeOptimizer',
    'FP16GradientAccumulator',
    'FP16TrainingMonitor',
    'fp16_clamp',
    'stable_softmax_fp16',
    'compute_loss_fp16_safe',
    'check_model_health',
    'fp16_autocast_context',
    'create_fp16_optimizer',
]