"""
FP16 Integration Module for Qwen Image Training

This module provides integration utilities to seamlessly incorporate fp16
training into the existing Qwen image training pipeline.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import logging
from contextlib import nullcontext
from musubi_tuner.qwen_image import qwen_image_model
from .fp16_utils import (
    DynamicLossScaler,
    FP16MasterWeights,
    FP16SafeOptimizer,
    FP16GradientAccumulator,
    FP16TrainingMonitor,
    compute_loss_fp16_safe,
    check_model_health,
    fp16_autocast_context,
    create_fp16_optimizer
)

logger = logging.getLogger(__name__)


class FP16TrainingManager:
    """
    Manages all aspects of fp16 training for Qwen image models.
    
    This class coordinates loss scaling, master weights, gradient accumulation,
    and health monitoring for stable fp16 training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: Any,
        accelerator: Any = None
    ):
        """
        Initialize FP16 training manager.
        
        Args:
            model: The model to train
            args: Training arguments
            accelerator: Optional Accelerate accelerator
        """
        self.model = model
        self.args = args
        self.accelerator = accelerator
        
        # Determine if we should use fp16 features
        self.use_fp16 = args.mixed_precision == "fp16"
        self.use_master_weights = self.use_fp16 and getattr(args, 'fp16_master_weights', True)
        self.use_loss_scaling = self.use_fp16 and getattr(args, 'fp16_loss_scaling', True)
        
        # Initialize components if using fp16
        if self.use_fp16:
            self._init_fp16_components()
        else:
            self.master_weights = None
            self.loss_scaler = None
            self.gradient_accumulator = None
            self.monitor = None
    
    def _init_fp16_components(self):
        """Initialize fp16-specific components."""
        # Master weights manager
        if self.use_master_weights:
            logger.info("Initializing fp32 master weights for fp16 training")
            self.master_weights = FP16MasterWeights(
                self.model,
                use_stochastic_rounding=getattr(self.args, 'fp16_stochastic_rounding', True),
                rng_seed=getattr(self.args, 'seed', None)
            )
        else:
            self.master_weights = None
        
        # Dynamic loss scaler
        if self.use_loss_scaling:
            logger.info("Initializing dynamic loss scaler for fp16 training")
            self.loss_scaler = DynamicLossScaler(
                init_scale=getattr(self.args, 'fp16_init_scale', 2**15),
                scale_factor=getattr(self.args, 'fp16_scale_factor', 2.0),
                scale_window=getattr(self.args, 'fp16_scale_window', 2000),
                min_scale=getattr(self.args, 'fp16_min_scale', 1.0),
                max_scale=getattr(self.args, 'fp16_max_scale', 2**24)
            )
        else:
            self.loss_scaler = None
        
        # Gradient accumulator
        accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        self.gradient_accumulator = FP16GradientAccumulator(
            accumulation_steps=accumulation_steps,
            loss_scaler=self.loss_scaler
        )
        
        # Training monitor
        self.monitor = FP16TrainingMonitor(
            log_interval=getattr(self.args, 'fp16_monitor_interval', 100)
        )
    
    def create_optimizer(
        self,
        model: Optional[nn.Module] = None,
        **optimizer_kwargs
    ) -> Union[FP16SafeOptimizer, torch.optim.Optimizer]:
        """
        Create optimizer configured for fp16 training.
        
        Args:
            model: Optional model (uses self.model if not provided)
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        model = model or self.model
        
        # Get optimizer configuration from args
        optimizer_type = getattr(self.args, 'optimizer_type', 'adamw')
        lr = getattr(self.args, 'learning_rate', 3e-4)
        weight_decay = getattr(self.args, 'weight_decay', 0.1)
        eps = getattr(self.args, 'adam_epsilon', 1e-8)
        betas = (
            getattr(self.args, 'adam_beta1', 0.9),
            getattr(self.args, 'adam_beta2', 0.999)
        )
        max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
        
        # Override with provided kwargs
        optimizer_kwargs = {
            'optimizer_type': optimizer_type,
            'lr': lr,
            'weight_decay': weight_decay,
            'eps': eps,
            'betas': betas,
            'max_grad_norm': max_grad_norm,
            'use_master_weights': self.use_master_weights,
            'use_loss_scaling': self.use_loss_scaling,
            **optimizer_kwargs
        }
        
        return create_fp16_optimizer(model, **optimizer_kwargs)
    
    def autocast_context(self):
        """Get appropriate autocast context for training."""
        if self.use_fp16:
            return fp16_autocast_context(
                device_type=self.get_device_type(),
                enabled=True
            )
        elif self.args.mixed_precision == "bf16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            return nullcontext()
    
    def get_device_type(self) -> str:
        """Get device type for autocast."""
        if self.accelerator and hasattr(self.accelerator, 'device'):
            return self.accelerator.device.type
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if self.gradient_accumulator:
            return self.gradient_accumulator.accumulate_loss(loss)
        return loss
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: str = "mse",
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Compute loss with fp16 safety."""
        if self.use_fp16:
            return compute_loss_fp16_safe(predictions, targets, loss_fn, reduction)
        else:
            # Standard loss computation
            if loss_fn == "mse":
                return torch.nn.functional.mse_loss(predictions, targets, reduction=reduction)
            elif loss_fn == "l1":
                return torch.nn.functional.l1_loss(predictions, targets, reduction=reduction)
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def optimizer_step(
        self,
        optimizer: Union[FP16SafeOptimizer, torch.optim.Optimizer],
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        Perform optimizer step with fp16 safety.
        
        Args:
            optimizer: The optimizer
            model: Optional model (uses self.model if not provided)
            
        Returns:
            True if step was successful, False if skipped
        """
        model = model or self.model
        
        # Check if we should update (gradient accumulation)
        if self.gradient_accumulator and not self.gradient_accumulator.should_update():
            return False
        
        # Perform optimizer step
        if isinstance(optimizer, FP16SafeOptimizer):
            success = optimizer.step(model)
        else:
            # Standard optimizer step
            if self.loss_scaler:
                # Unscale gradients for standard optimizer
                self.loss_scaler.unscale_gradients(optimizer)
                
                # Check for overflow
                has_overflow, has_underflow = self.loss_scaler.check_gradients(model.parameters())
                
                if not has_overflow:
                    # Clip gradients
                    max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    success = True
                else:
                    success = False
                
                # Update loss scale
                self.loss_scaler.update_scale(has_overflow, has_underflow)
            else:
                # No loss scaling
                optimizer.step()
                success = True
        
        # Zero gradients
        optimizer.zero_grad()
        
        return success
    
    def check_model_health(self) -> Dict[str, Any]:
        """Check model health and return diagnostics."""
        return check_model_health(self.model)
    
    def log_step(
        self,
        loss: torch.Tensor,
        grad_norm: Optional[float] = None
    ):
        """Log training step for monitoring."""
        if self.monitor and self.use_fp16:
            if grad_norm is None:
                grad_norm = self._compute_grad_norm()
            
            loss_scale = self.loss_scaler.scale if self.loss_scaler else 1.0
            has_overflow = False
            has_underflow = False
            
            if self.loss_scaler:
                has_overflow = self.loss_scaler._overflow_count > 0
                has_underflow = self.loss_scaler._underflow_count > 0
            
            self.monitor.log_step(
                loss=loss,
                grad_norm=grad_norm,
                loss_scale=loss_scale,
                has_overflow=has_overflow,
                has_underflow=has_underflow
            )
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm of model."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        state = {}
        
        if self.master_weights:
            state['master_weights'] = self.master_weights.state_dict()
        
        if self.loss_scaler:
            state['loss_scaler'] = self.loss_scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint."""
        if self.master_weights and 'master_weights' in state_dict:
            self.master_weights.load_state_dict(state_dict['master_weights'])
        
        if self.loss_scaler and 'loss_scaler' in state_dict:
            self.loss_scaler.load_state_dict(state_dict['loss_scaler'])


def modify_args_for_fp16(args) -> None:
    """
    Modify training arguments to enable safe fp16 training.
    
    This function adjusts various training parameters to be more suitable
    for fp16 training, following recommendations from llm.c.
    
    Args:
        args: Training arguments to modify (modified in-place)
    """
    if args.mixed_precision != "fp16":
        return
    
    logger.info("Configuring arguments for fp16 training")
    
    # Enable fp16-specific features
    if not hasattr(args, 'fp16_master_weights'):
        args.fp16_master_weights = True
        logger.info("Enabled fp32 master weights for fp16 training")
    
    if not hasattr(args, 'fp16_loss_scaling'):
        args.fp16_loss_scaling = True
        logger.info("Enabled dynamic loss scaling for fp16 training")
    
    if not hasattr(args, 'fp16_stochastic_rounding'):
        args.fp16_stochastic_rounding = True
        logger.info("Enabled stochastic rounding for fp16 training")
    
    # Adjust learning rate if needed (fp16 often needs lower LR)
    if hasattr(args, 'learning_rate') and args.learning_rate > 1e-4:
        original_lr = args.learning_rate
        args.learning_rate = min(args.learning_rate, 5e-5)
        if args.learning_rate != original_lr:
            logger.info(f"Reduced learning rate from {original_lr} to {args.learning_rate} for fp16 stability")
    
    # Set conservative loss scaling parameters
    if not hasattr(args, 'fp16_init_scale'):
        args.fp16_init_scale = 2**15  # Conservative starting scale
    
    if not hasattr(args, 'fp16_scale_window'):
        args.fp16_scale_window = 2000  # How often to increase scale
    
    # Enable gradient clipping if not set
    if not hasattr(args, 'max_grad_norm') or args.max_grad_norm <= 0:
        args.max_grad_norm = 1.0
        logger.info("Enabled gradient clipping with max_norm=1.0 for fp16 training")
    
    # Adjust epsilon for Adam optimizer
    if hasattr(args, 'adam_epsilon') and args.adam_epsilon < 1e-7:
        args.adam_epsilon = 1e-7
        logger.info(f"Increased adam_epsilon to {args.adam_epsilon} for fp16 stability")


def create_fp16_compatible_model(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Prepare model for fp16 training.
    
    This function:
    1. Converts model to fp16
    2. Keeps certain layers in fp32 for stability (BatchNorm, LayerNorm)
    3. Applies other fp16-specific modifications
    
    Args:
        model: Model to prepare
        dtype: Target dtype (should be torch.float16)
        
    Returns:
        Model prepared for fp16 training
    """
    if dtype != torch.float16:
        return model
    
    logger.info("Preparing model for fp16 training")
    
    # Convert model to fp16
    model = model.to(dtype)
    
    # Keep normalization layers in fp32 for stability
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm, qwen_image_model.RMSNorm,
                              qwen_image_model.AdaLayerNormContinuous, qwen_image_model.AdaLayerNorm)):
            module.float()
            logger.debug(f"Keeping {module.__class__.__name__} in fp32 for stability")
    
    return model


class FP16TrainingWrapper:
    """
    Context manager for fp16 training that handles all setup and teardown.
    
    Usage:
        with FP16TrainingWrapper(model, args) as fp16_manager:
            optimizer = fp16_manager.create_optimizer()
            for batch in dataloader:
                with fp16_manager.autocast_context():
                    outputs = model(batch)
                    loss = fp16_manager.compute_loss(outputs, targets)
                
                scaled_loss = fp16_manager.scale_loss(loss)
                scaled_loss.backward()
                
                fp16_manager.optimizer_step(optimizer)
                fp16_manager.log_step(loss)
    """
    
    def __init__(self, model: nn.Module, args: Any):
        """
        Initialize fp16 training wrapper.
        
        Args:
            model: Model to train
            args: Training arguments
        """
        self.model = model
        self.args = args
        self.manager = None
        self.original_dtype = None
    
    def __enter__(self):
        """Setup fp16 training."""
        # Modify args for fp16
        modify_args_for_fp16(self.args)
        
        # Store original dtype
        self.original_dtype = next(self.model.parameters()).dtype
        
        # Prepare model for fp16
        if self.args.mixed_precision == "fp16":
            self.model = create_fp16_compatible_model(self.model, torch.float16)
        
        # Create training manager
        self.manager = FP16TrainingManager(self.model, self.args)
        
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup after fp16 training."""
        # Restore original dtype if needed
        if self.original_dtype and self.original_dtype != torch.float16:
            self.model = self.model.to(self.original_dtype)
        
        # Log final statistics
        if self.manager and self.manager.monitor:
            self.manager.monitor.print_summary()


# Export main components
__all__ = [
    'FP16TrainingManager',
    'FP16TrainingWrapper',
    'modify_args_for_fp16',
    'create_fp16_compatible_model',
]