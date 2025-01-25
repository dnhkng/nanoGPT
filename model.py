"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

def get_default_activation_config():
    """Factory function to create default activation config"""
    return ActivationConfig()

@dataclass
class ActivationConfig:
    """Configuration for activation functions."""
    name: str = 'softmax'  # One of: 'softmax', 'improved_bounded'
    temperature: float = 1.0  # Temperature parameter for controlling distribution sharpness
    scale_factor: float = 1.0  # Base scaling factor for logits
    min_temp: float = 1.0  # Minimum temperature for adaptive temperature functions
    alpha: float = 0.3  # Initial mixing parameter

    def get_activation(self):
        """Get the activation function instance based on config."""
        if self.name == 'softmax':
            return SoftmaxActivation()
        elif self.name == 'improved_bounded':
            return ImprovedBoundedActivation(
                temperature=self.temperature,
                scale_factor=self.scale_factor
            )
        elif self.name == 'fast_bounded':
            return FastBoundedActivation(self.temperature)
        elif self.name == 'logistic_bounded':
            return LogisticBoundedActivation()
        elif self.name == 'logistic_bounded_root':
            return LogisticBoundedActivationRoot()
        elif self.name == 'logistic_bounded_arctan':
            return LogisticBoundedActivationArcTan()
        elif self.name == 'logistic_bounded_relu':
            return LogisticBoundedActivationReLU()
        elif self.name == 'logistic_bounded_dynamic':
            return LogisticBoundedActivationDynamic()
        elif self.name == 'logistic_bounded_percentile':
            return LogisticBoundedActivationPercentile()
        elif self.name == 'logistic_bounded_softplus':
            return LogisticBoundedActivationSoftPlus()
        elif self.name == 'logistic_bounded_hybrid':
            return LogisticBoundedActivationHybrid()
        elif self.name == 'scale_exp':
            return ExpScaledActivation(self.temperature)
        elif self.name == 'scale_smooth':
            return SmoothScaledActivation(self.temperature)
        elif self.name == 'scale_minmax':
            return MinMaxScaledActivation(self.temperature)
        elif self.name == 'scale_harmonic':
            return HarmonicScaledActivation(self.temperature)
        elif self.name == 'kl_bounded':
            return KLBoundedActivation(self.temperature)
        elif self.name == 'optimized':
            return OptimizedActivation(self.temperature)
        elif self.name == 'optimized_v2':
            return OptimizedActivationV2(self.temperature)
        elif self.name == 'simple_bounded':
            return SimpleBoundedActivation(self.temperature)
        elif self.name == 'hybrid_bounded':
            return HybridBoundedActivation(self.temperature)
        elif self.name == 'enhanced_simple_bounded':
            return EnhancedSimpleBoundedActivation(self.temperature)
        elif self.name == 'hybrid_optimized':
            return HybridOptimizedActivation(self.temperature, self.scale_factor)
        elif self.name == 'optimized_simple_bounded':
            return OptimizedSimpleBoundedActivation(self.temperature)
        elif self.name == 'refined_simple_bounded':
            return RefinedSimpleBoundedActivation(self.temperature)
        elif self.name == 'final_simple_bounded':
            return FinalSimpleBoundedActivation(self.temperature, self.min_temp)
        elif self.name == 'enhanced_relu_bounded':
            return EnhancedReLUBoundedActivation()
        elif self.name == 'adaptive_relu_bounded':
            return AdaptiveReLUBoundedActivation()
        elif self.name == 'hybrid_relu_bounded':
            return HybridReLUBoundedActivation()
        elif self.name == 'hybrid_relu_bounded_ce_loss':
            return HybridReLUBoundedActivationCELoss()
        elif self.name == 'gradient_stable_relu_bounded':
            return GradientStableReLUBoundedActivation()
        elif self.name == 'dynamic_hybrid': #next
            return DynamicHybridActivation(self.temperature)
        elif self.name == 'dynamic_hybrid_ce_loss':
            return DynamicHybridActivationCELoss(self.temperature)
        elif self.name == 'gated_hybrid':
            return GatedHybridActivation()
        elif self.name == 'residual_hybrid':
            return ResidualHybridActivation()
        elif self.name == 'adaptive_gating_hybrid':
            return AdaptiveGatingHybridActivation()
        elif self.name == 'scale_invariant_hybrid':
            return ScaleInvariantHybridActivation()
        elif self.name == 'enhanced_gradient_stable_relu':
            return EnhancedGradientStableReLUActivation(self.scale_factor)
        elif self.name == 'adaptive_gradient_stable_relu':
            return AdaptiveGradientStableReLUActivation(self.scale_factor)
        elif self.name == 'hybrid_gradient_stable_relu':
            return HybridGradientStableReLUActivation(self.scale_factor)
        elif self.name == 'residual_gradient_stable_relu':
            return ResidualGradientStableReLUActivation(self.scale_factor)
        elif self.name == 'scale_aware_gradient_stable_relu':
            return ScaleAwareGradientStableReLUActivation(self.temperature)
        elif self.name == 'dynamic_gradient_stable_relu':
            return DynamicGradientStableReLUActivation(self.scale_factor)
        elif self.name == 'dynamic_hybrid_activation2':
            return DynamicHybridActivation2(self.alpha, self.temperature)
        elif self.name == 'residual_hybrid_activation2':
            return ResidualHybridActivation2(self.alpha)
        elif self.name == 'hybrid_relu_bounded_activation2':
            return HybridReLUBoundedActivation2(self.alpha)
        elif self.name == 'gated_hybrid_activation2':
            return GatedHybridActivation2(self.alpha)
        elif self.name == 'dynamic_hybrid_activation_optimised':
            return DynamicHybridActivationOptimised(self.alpha, self.temperature)
        else:
            raise ValueError(f"Unknown activation: {self.name}")

import torch
import torch.nn as nn
import torch.nn.functional as F



class DynamicHybridActivationOptimised(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('temperature', torch.tensor(temperature))
        self.eps = 1e-6
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=-100, max=100)  # Prevent extreme values
        std = torch.clamp(torch.std(x, dim=-1, keepdim=True), min=self.eps)
        temp = torch.clamp(self.temperature / (1 + torch.log1p(std)), min=self.eps)
        
        # Safe softmax
        x_scaled = x / temp
        x_max = x_scaled.max(dim=-1, keepdim=True)[0]
        exp_x = torch.exp(torch.clamp(x_scaled - x_max, min=-15, max=15))
        denom = torch.clamp(exp_x.sum(dim=-1, keepdim=True), min=self.eps)
        soft_probs = exp_x / denom
        
        # Bounded component
        scale = torch.log1p(F.relu(x)) + self.eps
        bounded = torch.tanh(x / scale)
        relu_probs = (bounded + 1) / 2
        
        # Safe entropy calculation
        log_soft = torch.log(soft_probs + self.eps)
        entropy = torch.clamp(-(soft_probs * log_soft).sum(dim=-1, keepdim=True), min=-20, max=20)
        dynamic_alpha = self.alpha * torch.sigmoid(-entropy)
        
        combined = dynamic_alpha * relu_probs + (1 - dynamic_alpha) * soft_probs
        return combined / torch.clamp(combined.sum(dim=-1, keepdim=True), min=self.eps)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        log_probs = torch.log(torch.clamp(probs, min=self.eps))
        
        loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            denom = torch.clamp(mask.sum(), min=self.eps)
            return (loss * mask).sum() / denom
            
        return loss.mean()


class OutputActivation(nn.Module):
    """Base class for output activation functions and loss computation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform logits into probability distribution."""
        raise NotImplementedError
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        """Compute loss between predictions and targets.
        
        Args:
            logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
            targets: Target indices of shape (batch_size, seq_len)
            ignore_index: Index to ignore in loss computation (default: -1)
            
        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities using the activation function
        probs = self.forward(logits)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        # Create one-hot encoded targets
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Compute cross entropy loss
        loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        
        return loss.mean()


class DynamicHybridActivation2(OutputActivation):
    """Dynamic hybrid activation that adapts based on input statistics."""
    
    def __init__(self, alpha: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adaptive temperature based on input scale
        temp = self.temperature / (1 + torch.log1p(torch.std(x, dim=-1, keepdim=True)))
        
        # Softmax with adaptive temperature
        soft_probs = F.softmax(x / temp, dim=-1)
        
        # ReLU bounded component
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Dynamic mixing based on prediction entropy
        entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1, keepdim=True)
        dynamic_alpha = self.alpha * torch.sigmoid(-entropy)
        
        combined = dynamic_alpha * relu_probs + (1 - dynamic_alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)


class ResidualHybridActivation2(OutputActivation):
    """Hybrid activation with residual connections between components."""
    
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base softmax
        soft_probs = F.softmax(x, dim=-1)
        
        # ReLU residual path
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Residual connection
        residual = self.alpha * (relu_probs - soft_probs)
        combined = soft_probs + residual
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)


class HybridReLUBoundedActivation2(OutputActivation):
    """Simple hybrid approach combining ReLU-based normalization with softmax."""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU-based component
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Softmax component
        soft_probs = F.softmax(x, dim=-1)
        
        # Combine both paths
        combined = self.alpha * relu_probs + (1 - self.alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)


class GatedHybridActivation2(OutputActivation):
    """Hybrid activation with learned gating between components."""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard softmax path
        soft_probs = F.softmax(x, dim=-1)
        
        # Gated ReLU path
        gate = torch.sigmoid(torch.max(x, dim=-1, keepdim=True)[0])
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Gated combination
        combined = gate * (self.alpha * relu_probs + (1 - self.alpha) * soft_probs) + \
                  (1 - gate) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)
    






class EnhancedGradientStableReLUActivation(OutputActivation):
    def __init__(self, scale_factor=0.3):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enhanced gradient-stable scaling
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10
        scale = F.relu(x) / norm * self.scale_factor
        scale = torch.log1p(scale) * torch.tanh(norm)
        
        bounded = torch.tanh(x / (1 + scale))
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    #     probs = self.forward(logits)
    #     probs = torch.clamp(probs, min=1e-10, max=1.0)
        
    #     target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
    #     # Cross entropy with gradient stability regularization
    #     ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
    #     norm = torch.norm(logits, dim=-1)
    #     reg_loss = 0.01 * torch.log1p(norm)
        
    #     loss = ce_loss + reg_loss
        
    #     if ignore_index >= 0:
    #         mask = (targets != ignore_index)
    #         loss = loss * mask
    #         return loss.sum() / (mask.sum() + 1e-10)
    #     return loss.mean()

class AdaptiveGradientStableReLUActivation(OutputActivation):
    def __init__(self, base_scale=0.5):
        super().__init__()
        self.base_scale = base_scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True) + 1e-10
        
        normalized_x = (x - mean) / std
        scale = F.relu(normalized_x) / (torch.norm(normalized_x, dim=-1, keepdim=True) + 1e-10)
        scale = torch.log1p(scale * self.base_scale)
        
        bounded = torch.tanh(normalized_x / (1 + scale))
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    #     probs = self.forward(logits)
    #     probs = torch.clamp(probs, min=1e-10, max=1.0)
        
    #     target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
    #     # Adaptive cross entropy with normalization penalty
    #     ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
    #     std = torch.std(logits, dim=-1)
    #     reg_loss = 0.01 * torch.log1p(std)
        
    #     loss = ce_loss + reg_loss
        
    #     if ignore_index >= 0:
    #         mask = (targets != ignore_index)
    #         loss = loss * mask
    #         return loss.sum() / (mask.sum() + 1e-10)
    #     return loss.mean()

class HybridGradientStableReLUActivation(OutputActivation):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10
        scale = F.relu(x) / norm
        bounded_stable = torch.tanh(x / (1 + torch.log1p(scale)))
        
        scale_relu = torch.log1p(F.relu(x))
        bounded_relu = torch.tanh(x / (1 + scale_relu))
        
        bounded = self.alpha * bounded_stable + (1 - self.alpha) * bounded_relu
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    #     probs = self.forward(logits)
    #     probs = torch.clamp(probs, min=1e-10, max=1.0)
        
    #     target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
    #     # Hybrid loss with balanced regularization
    #     ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
    #     norm = torch.norm(logits, dim=-1)
    #     relu_term = torch.mean(F.relu(logits), dim=-1)
    #     reg_loss = 0.01 * (self.alpha * torch.log1p(norm) + (1 - self.alpha) * torch.log1p(relu_term))
        
    #     loss = ce_loss + reg_loss
        
    #     if ignore_index >= 0:
    #         mask = (targets != ignore_index)
    #         loss = loss * mask
    #         return loss.sum() / (mask.sum() + 1e-10)
    #     return loss.mean()

class ResidualGradientStableReLUActivation(OutputActivation):
    def __init__(self, scale_factor=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10
        base_scale = F.relu(x) / norm
        base_bounded = torch.tanh(x / (1 + torch.log1p(base_scale)))
        
        residual_scale = torch.log1p(F.relu(x)) * self.scale_factor
        residual = torch.tanh(x / (1 + residual_scale))
        
        bounded = base_bounded + residual_scale * (residual - base_bounded)
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    #     probs = self.forward(logits)
    #     probs = torch.clamp(probs, min=1e-10, max=1.0)
        
    #     target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
    #     # Residual-aware loss
    #     ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
    #     residual_term = torch.norm(F.relu(logits), dim=-1)
    #     reg_loss = 0.01 * self.scale_factor * torch.log1p(residual_term)
        
    #     loss = ce_loss + reg_loss
        
    #     if ignore_index >= 0:
    #         mask = (targets != ignore_index)
    #         loss = loss * mask
    #         return loss.sum() / (mask.sum() + 1e-10)
    #     return loss.mean()

class DynamicGradientStableReLUActivation(OutputActivation):
    def __init__(self, base_scale=0.5, adapt_rate=0.1):
        super().__init__()
        self.base_scale = base_scale
        self.adapt_rate = adapt_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        
        adapt_scale = torch.sigmoid(self.adapt_rate * (max_val - norm))
        scale = F.relu(x) / norm * (self.base_scale + adapt_scale)
        
        scale = torch.log1p(scale) * torch.tanh(norm)
        bounded = torch.tanh(x / (1 + scale))
        
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    #     probs = self.forward(logits)
    #     probs = torch.clamp(probs, min=1e-10, max=1.0)
        
    #     target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
    #     # Dynamic adaptation-aware loss
    #     ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        
    #     norm = torch.norm(logits, dim=-1)
    #     max_val = torch.max(logits, dim=-1)[0]
    #     adapt_term = torch.sigmoid(self.adapt_rate * (max_val - norm))
    #     reg_loss = 0.01 * (self.base_scale * torch.log1p(norm) + adapt_term)
        
    #     loss = ce_loss + reg_loss
        
    #     if ignore_index >= 0:
    #         mask = (targets != ignore_index)
    #         loss = loss * mask
    #         return loss.sum() / (mask.sum() + 1e-10)
    #     return loss.mean()


class ScaleAwareGradientStableReLUActivation(OutputActivation):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale-aware normalization
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-10
        scale_factor = torch.tanh(norm / self.temperature)
        
        # Gradient-stable scaling with scale awareness
        scale = F.relu(x) / norm
        scale = torch.log1p(scale) * scale_factor
        
        # Additional stability for large magnitudes
        magnitude_gate = torch.sigmoid(norm - self.temperature)
        safe_scale = scale * (1 - magnitude_gate) + magnitude_gate
        
        bounded = torch.tanh(x / (1 + safe_scale))
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    #     probs = self.forward(logits)
    #     probs = torch.clamp(probs, min=1e-10, max=1.0)
        
    #     target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
    #     # Scale-aware cross entropy with temperature-based regularization
    #     ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        
    #     # Scale-aware regularization
    #     norm = torch.norm(logits, dim=-1)
    #     scale_factor = torch.tanh(norm / self.temperature)
    #     magnitude_gate = torch.sigmoid(norm - self.temperature)
        
    #     # Combined regularization term that considers both scale and magnitude
    #     reg_loss = 0.01 * (
    #         scale_factor * torch.log1p(norm) +  # Scale-dependent term
    #         magnitude_gate * torch.abs(norm - self.temperature)  # Magnitude penalty
    #     )
        
    #     loss = ce_loss + reg_loss
        
    #     if ignore_index >= 0:
    #         mask = (targets != ignore_index)
    #         loss = loss * mask
    #         return loss.sum() / (mask.sum() + 1e-10)
    #     return loss.mean()


class DynamicHybridActivationCELoss(OutputActivation):
    def __init__(self, alpha=0.5, temperature=1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adaptive temperature based on input scale
        temp = self.temperature / (1 + torch.log1p(torch.std(x, dim=-1, keepdim=True)))
        
        # Softmax with adaptive temperature
        soft_probs = F.softmax(x / temp, dim=-1)
        
        # ReLU bounded component
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Dynamic mixing based on prediction entropy
        entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1, keepdim=True)
        dynamic_alpha = self.alpha * torch.sigmoid(-entropy)
        
        combined = dynamic_alpha * relu_probs + (1 - dynamic_alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        # Stabilize probabilities
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        # Create one-hot encoded targets
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Compute cross entropy with stability terms
        loss = -torch.sum(target_one_hot * torch.log(probs + 1e-10), dim=-1)

        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        
        return loss.mean()


class DynamicHybridActivation(OutputActivation):
    def __init__(self, alpha=0.5, temperature=1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adaptive temperature based on input scale
        temp = self.temperature / (1 + torch.log1p(torch.std(x, dim=-1, keepdim=True)))
        
        # Softmax with adaptive temperature
        soft_probs = F.softmax(x / temp, dim=-1)
        
        # ReLU bounded component
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Dynamic mixing based on prediction entropy
        entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1, keepdim=True)
        dynamic_alpha = self.alpha * torch.sigmoid(-entropy)
        
        combined = dynamic_alpha * relu_probs + (1 - dynamic_alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        # Stabilize probabilities
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        # Create one-hot encoded targets
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Compute cross entropy with stability terms
        loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        
        # Add small entropy regularization
        entropy_reg = 0.01 * torch.sum(probs * torch.log(probs), dim=-1)
        loss = loss + entropy_reg
        
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        
        return loss.mean()

class GatedHybridActivation(OutputActivation):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard softmax path
        soft_probs = F.softmax(x, dim=-1)
        
        # Gated ReLU path
        gate = torch.sigmoid(torch.max(x, dim=-1, keepdim=True)[0])
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Gated combination
        combined = gate * (self.alpha * relu_probs + (1 - self.alpha) * soft_probs) + \
                  (1 - gate) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()
    
class ResidualHybridActivation(OutputActivation):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base softmax
        soft_probs = F.softmax(x, dim=-1)
        
        # ReLU residual
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Residual connection
        residual = self.alpha * (relu_probs - soft_probs)
        combined = soft_probs + residual
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Cross entropy with residual regularization
        ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        reg_loss = 0.01 * torch.norm(probs - F.softmax(logits, dim=-1), dim=-1)
        loss = ce_loss + reg_loss
        
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()
    
class AdaptiveGatingHybridActivation(OutputActivation):
    def __init__(self, alpha=0.5, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute statistics
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        
        # Adaptive softmax
        soft_probs = F.softmax(x / (1 + self.beta * std), dim=-1)
        
        # Adaptive ReLU
        scale = torch.log1p(F.relu(x - mean))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Adaptive gating
        gate = torch.sigmoid((x - mean) / (std + 1e-10))
        combined = self.alpha * gate * relu_probs + (1 - self.alpha * gate) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Cross entropy with adaptive regularization
        ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        
        # Add stability term based on gating
        mean = logits.mean(dim=-1, keepdim=True)
        std = torch.std(logits, dim=-1, keepdim=True)
        gate = torch.sigmoid((logits - mean) / (std + 1e-10))
        reg_loss = 0.01 * torch.sum(gate * torch.log(gate + 1e-10), dim=-1)
        
        loss = ce_loss + reg_loss
        
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class ScaleInvariantHybridActivation(OutputActivation):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
        
        # Scale-invariant softmax
        soft_probs = F.softmax(x_norm, dim=-1)
        
        # Scale-aware ReLU
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x_norm / (1 + torch.tanh(scale)))
        relu_probs = (bounded + 1) / 2
        
        # Combine with scale-dependent mixing
        scale_factor = torch.tanh(torch.norm(x, dim=-1, keepdim=True))
        dynamic_alpha = self.alpha * scale_factor
        
        combined = dynamic_alpha * relu_probs + (1 - dynamic_alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Scale-aware cross entropy
        norm = torch.norm(logits, dim=-1, keepdim=True)
        scale_factor = torch.tanh(norm)
        
        ce_loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        scale_reg = 0.01 * scale_factor.squeeze(-1) * torch.sum(probs * torch.log(probs), dim=-1)
        
        loss = ce_loss + scale_reg
        
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class EnhancedReLUBoundedActivation(OutputActivation):
    """
    Enhanced ReLU-based activation with KL-like loss.
    Uses separate positive/negative scaling.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Separate scaling for positive and negative values
        pos_scale = torch.log1p(F.relu(x))
        neg_scale = torch.log1p(F.relu(-x))
        scale = pos_scale + neg_scale
        
        # Balanced activation
        bounded = torch.tanh(x / (1 + scale))
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        target_dist = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # KL divergence terms
        kl_loss = target_dist * (torch.log(target_dist + 1e-10) - torch.log(probs + 1e-10))
        # Add regularization term
        reg_term = -0.1 * torch.mean(probs * torch.log(probs + 1e-10), dim=-1)
        
        loss = kl_loss.sum(dim=-1) + reg_term
        
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class AdaptiveReLUBoundedActivation(OutputActivation):
    """
    Adaptive ReLU-based activation with temperature scaling
    """
    def __init__(self, base_temp=2.0):
        super().__init__()
        self.base_temp = base_temp
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute adaptive temperature
        temp = self.base_temp / (1 + torch.std(x, dim=-1, keepdim=True))
        
        # Dynamic scaling based on ReLU
        scale = torch.log1p(F.relu(x/temp))
        bounded = torch.tanh(x / (1 + scale))
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        target_dist = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Symmetric KL divergence
        forward_kl = torch.sum(target_dist * (torch.log(target_dist + 1e-10) - torch.log(probs + 1e-10)), dim=-1)
        reverse_kl = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(target_dist + 1e-10)), dim=-1)
        loss = (forward_kl + reverse_kl) / 2
        
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class HybridReLUBoundedActivationCELoss(OutputActivation):
    """
    Hybrid approach combining ReLU with softmax-like behavior
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU-based component
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Softmax-like component
        soft_probs = F.softmax(x, dim=-1)
        
        # Combine both
        combined = self.alpha * relu_probs + (1 - self.alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        # Stabilize probabilities
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        # Create one-hot encoded targets
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Compute cross entropy with stability terms
        # loss = -torch.sum(target_one_hot * torch.log(probs), dim=-1)
        loss = -torch.sum(target_one_hot * torch.log(probs + 1e-10), dim=-1)

        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        
        return loss.mean()



class HybridReLUBoundedActivation(OutputActivation):
    """
    Hybrid approach combining ReLU with softmax-like behavior
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU-based component
        scale = torch.log1p(F.relu(x))
        bounded = torch.tanh(x / (1 + scale))
        relu_probs = (bounded + 1) / 2
        
        # Softmax-like component
        soft_probs = F.softmax(x, dim=-1)
        
        # Combine both
        combined = self.alpha * relu_probs + (1 - self.alpha) * soft_probs
        return combined / (combined.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        target_dist = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # KL divergence with entropy regularization
        kl_loss = target_dist * (torch.log(target_dist + 1e-10) - torch.log(probs + 1e-10))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        loss = kl_loss.sum(dim=-1) - 0.1 * entropy
        
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class GradientStableReLUBoundedActivation(OutputActivation):
    """
    ReLU-based activation with gradient stabilization
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient-stable scaling
        scale = F.relu(x) / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
        scale = torch.log1p(scale)
        
        bounded = torch.tanh(x / (1 + scale))
        probs = (bounded + 1) / 2
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        target_dist = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Stabilized KL divergence
        eps = 1e-10
        probs_stable = torch.clamp(probs, eps, 1.0 - eps)
        target_stable = torch.clamp(target_dist, eps, 1.0 - eps)
        
        kl_div = target_stable * (torch.log(target_stable) - torch.log(probs_stable))
        loss = kl_div.sum(dim=-1)
        
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss * mask
            return loss.sum() / (mask.sum() + eps)
        return loss.mean()

class OptimizedSimpleBoundedActivation(OutputActivation):
    def __init__(self, temperature=2.0, min_temp=1.5):
        super().__init__()
        self.temperature = temperature
        self.min_temp = min_temp
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adaptive temperature based on input scale
        scale = torch.std(x, dim=-1, keepdim=True)
        temp = torch.maximum(
            self.temperature / (1 + torch.log1p(scale)),
            torch.tensor(self.min_temp)
        )
        
        # Core activation
        activated = torch.sigmoid(x / temp)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)


    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()


class FinalSimpleBoundedActivation(OutputActivation):
    def __init__(self, temperature=2.0, min_temp=1.5):
        super().__init__()
        self.temperature = temperature
        self.min_temp = min_temp
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adaptive temperature based on input scale
        scale = torch.std(x, dim=-1, keepdim=True)
        temp = torch.maximum(
            self.temperature / (1 + torch.log1p(scale)),
            torch.tensor(self.min_temp, device=x.device)
        )
        
        # Core activation
        activated = torch.sigmoid(x / temp)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class SimpleBoundedActivation(OutputActivation):
    """Extremely simplified version"""
    
    def __init__(self, temperature=2.0):  # Higher default temperature
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Just temperature scaling and sigmoid
        activated = torch.sigmoid(x / self.temperature)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()
    
class RefinedSimpleBoundedActivation(OutputActivation):
    def __init__(self, temperature=2.0, min_temp=1.5, scale_factor=0.1):
        super().__init__()
        self.temperature = temperature
        self.min_temp = min_temp
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute scale using exponential moving std
        with torch.no_grad():
            scale = torch.std(x, dim=-1, keepdim=True)
            scale = scale * self.scale_factor + (1 - self.scale_factor) * self.temperature
        
        # Bounded temperature
        temp = torch.clamp(
            scale,
            min=self.min_temp,
            max=self.temperature * 2.0
        )
        
        # Simple activation with bounded temperature
        activated = torch.sigmoid(x / temp)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()



class HybridOptimizedActivation(OutputActivation):
    def __init__(self, temperature=2.0, scale_factor=0.1):
        super().__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute adaptive scale based on input distribution
        mean_scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        std_scale = torch.std(x, dim=-1, keepdim=True)
        
        # Combine scales with learned weight
        adaptive_temp = self.temperature * (
            1.0 + self.scale_factor * torch.log1p(mean_scale + std_scale)
        )
        
        # Core activation with adaptive temperature
        activated = torch.sigmoid(x / adaptive_temp)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()

class EnhancedSimpleBoundedActivation(OutputActivation):
    def __init__(self, temperature=2.0, min_temp=1.5, max_temp=2.5):
        super().__init__()
        self.temperature = temperature
        self.min_temp = min_temp
        self.max_temp = max_temp
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic temperature based on input magnitude
        batch_temp = self.temperature * torch.clamp(
            1.0 + torch.std(x) * 0.1,  # Slight adjustment based on input variance
            self.min_temp/self.temperature,
            self.max_temp/self.temperature
        )
        
        # Core simple bounded activation
        activated = torch.sigmoid(x / batch_temp)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()
class HybridBoundedActivation(OutputActivation):
    """Combines successful elements from top performers"""
    
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale using arctan like the successful version
        scale = torch.arctan(torch.abs(x)) * 2/math.pi
        x_scaled = x / (1 + scale) / self.temperature
        
        # Simple sigmoid activation
        activated = torch.sigmoid(x_scaled)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        activated = self.forward(logits)
        # Create one-hot encoded target
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        # Compute cross entropy manually
        loss = -torch.sum(target_one_hot * torch.log(activated + 1e-10), dim=-1)
        # Handle ignore_index
        if ignore_index >= 0:
            mask = (targets != ignore_index)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-10)
        return loss.mean()


    
class OptimizedActivationV2(OutputActivation):
    """Simplified version focusing on core elements that worked well"""
    
    def __init__(self, temperature=1.0, label_smoothing=0.0):  # Disabled label smoothing by default
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simpler scaling more similar to softmax
        x_scaled = x / self.temperature
        
        # Direct sigmoid without extra scaling
        activated = torch.sigmoid(x_scaled)
        
        # Normalize
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        
        # Simple cross entropy without label smoothing
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        
        # Standard cross entropy
        loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-10), dim=-1)
        
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        
        return loss.mean()

class OptimizedActivation(OutputActivation):
    """Combines the best aspects of the top performing activations"""
    
    def __init__(self, temperature=1.0, label_smoothing=0.1):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale using a simple, effective approach
        scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        x_scaled = x / (scale + 1e-6) * self.temperature
        
        # Use sigmoid for bounded activation
        activated = torch.sigmoid(x_scaled)
        
        # Simple, numerically stable normalization
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        
        # Convert targets to one-hot with label smoothing
        num_classes = logits.size(-1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        
        # Apply label smoothing
        smooth_targets = (1.0 - self.label_smoothing) * targets_one_hot + \
                        self.label_smoothing / num_classes
        
        # Compute cross entropy with improved numerical stability
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        
        return loss.mean()

class KLBoundedActivation(OutputActivation):
    """
    Bounded activation that aims to preserve KL divergence properties.
    Uses a transformation that maintains relative differences between logits
    while bounding the output range.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Center logits to improve numerical stability and 
        # preserve relative differences
        x = x - x.mean(dim=-1, keepdim=True)
        
        # Scale logits to control the range while preserving ratios
        scale = torch.log1p(torch.exp(torch.abs(x)))
        x_scaled = x / (scale + 1e-6) * self.temperature
        
        # Transform to (0,1) range while preserving relative differences
        activated = torch.exp(x_scaled) / (1 + torch.exp(x_scaled))
        
        # Normalize to get proper probability distribution
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get predicted distribution
        probs = self.forward(logits)
        
        # Convert targets to one-hot
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        
        # Compute KL divergence style loss
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        
        return loss.mean()

class ExpScaledActivation(OutputActivation):
    """
    Uses exponential moving average of magnitudes for scaling.
    Should provide smooth transitions between scaling factors.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.momentum = 0.9
        self.register_buffer('running_scale', torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update running scale with exponential moving average
        with torch.no_grad():
            current_scale = torch.mean(torch.abs(x))
            self.running_scale = self.momentum * self.running_scale + (1 - self.momentum) * current_scale

        # Scale using running average
        x_scaled = x / (self.running_scale / self.temperature + 1e-5)
        activated = torch.sigmoid(x_scaled)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-10), dim=-1)
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class SmoothScaledActivation(OutputActivation):
    """
    Uses a smooth scaling function that combines local and global information.
    Should provide good scaling across different magnitude ranges.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combine local and global scaling
        local_scale = torch.abs(x)
        global_scale = torch.mean(local_scale, dim=-1, keepdim=True)
        
        # Smooth transition between scales
        alpha = torch.sigmoid(local_scale - global_scale)
        scale = alpha * local_scale + (1 - alpha) * global_scale
        
        x_scaled = x / (torch.log1p(scale) / self.temperature + 1e-5)
        activated = torch.sigmoid(x_scaled)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-10), dim=-1)
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class MinMaxScaledActivation(OutputActivation):
    """
    Uses min-max scaling within local windows for more adaptive scaling.
    Should handle varying ranges of values well.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate local min-max scaling
        x_min = torch.min(x, dim=-1, keepdim=True)[0]
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        x_range = x_max - x_min + 1e-5
        
        # Scale to [0, 1] range then apply temperature
        x_scaled = (x - x_min) / x_range * self.temperature
        activated = torch.sigmoid(x_scaled)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-10), dim=-1)
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class HarmonicScaledActivation(OutputActivation):
    """
    Uses harmonic mean for scaling, which should be less sensitive to outliers
    while still providing good scaling properties.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate harmonic mean-based scaling
        eps = 1e-5
        abs_x = torch.abs(x) + eps
        harmonic_scale = torch.reciprocal(
            torch.mean(torch.reciprocal(abs_x), dim=-1, keepdim=True)
        )
        
        x_scaled = x / (harmonic_scale / self.temperature + eps)
        activated = torch.sigmoid(x_scaled)
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + eps)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        probs = self.forward(logits)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-10), dim=-1)
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivation(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.log1p(torch.abs(x))
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationRoot(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.sqrt(torch.abs(x) + 1)
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationArcTan(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.arctan(torch.abs(x)) * 2/math.pi
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationReLU(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.log1p(F.relu(x))
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationDynamic(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationPercentile(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.quantile(torch.abs(x), 0.9, dim=-1, keepdim=True)
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationSoftPlus(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = F.softplus(torch.abs(x))
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class LogisticBoundedActivationHybrid(OutputActivation):
    """
    LogisticBounded activation with automatic normalization.
    Converts raw logits to probabilities using tanh with adaptive scaling.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the logits adaptively using log1p
        scale = torch.log1p(torch.abs(x)) * torch.tanh(torch.abs(x))
        # Apply bounded activation
        bounded = torch.tanh(x / (1 + scale))
        # Convert from [-1,1] to [0,1] and normalize
        probs = (bounded + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True)
    

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        # Calculate cross entropy loss using scatter
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        return loss.mean()

class ImprovedBoundedActivation(OutputActivation):
    """
    Enhanced bounded activation with better convergence properties.
    Key improvements:
    1. Adaptive temperature scaling based on input magnitude
    2. Improved normalization with better gradient flow
    3. Smoother bounded activation using a modified sigmoid
    4. Optional focal loss component for better handling of rare tokens
    """
    
    def __init__(self, temperature: float = 1.0, scale_factor: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        
    def _adaptive_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive scaling factor based on input distribution.
        Uses a combination of mean and max values to better handle different ranges.
        """
        mean_abs = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        max_abs = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        return self.scale_factor * torch.log1p(0.5 * (mean_abs + max_abs))

    def _bounded_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced bounded activation using a modified sigmoid function.
        Provides smoother gradients compared to tanh while maintaining boundedness.
        """
        scale = self._adaptive_scale(x)
        x_scaled = x / (scale + 1e-6)  # Prevent division by zero
        return torch.sigmoid(x_scaled / self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities with enhanced stability and gradient flow.
        """
        # Apply bounded activation
        activated = self._bounded_activation(x)
        
        # Improved normalization with gradient-friendly epsilon
        eps = 1e-7
        denom = torch.sum(activated + eps, dim=-1, keepdim=True)
        probs = activated / denom
        
        # Ensure strict probability distribution
        probs = torch.clamp(probs, min=eps, max=1.0)
        return probs / torch.sum(probs, dim=-1, keepdim=True)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        """
        Enhanced loss computation with better numerical stability and gradient properties.
        """
        # Get probabilities with temperature-scaled logits
        probs = self.forward(logits)
        
        # Convert targets to one-hot with label smoothing
        smooth_factor = 0.1  # Small smoothing factor for better generalization
        num_classes = logits.size(-1)
        
        # Create smoothed one-hot vectors
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        smooth_targets = (1.0 - smooth_factor) * targets_one_hot + \
                        smooth_factor / num_classes
        
        # Compute loss with improved numerical stability
        eps = 1e-7
        log_probs = torch.log(probs + eps)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        
        # Add regularization term to encourage confidence in predictions
        entropy_reg = -torch.sum(probs * log_probs, dim=-1)
        reg_weight = 0.01  # Small regularization weight
        
        return loss.mean() + reg_weight * entropy_reg.mean()


class FastBoundedActivation(OutputActivation):
    """
    Fast converging bounded activation that maintains good gradient flow.
    Uses a simpler, more direct approach for better convergence speed.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the input for better gradient flow
        x = x / self.temperature
        
        # Use sigmoid for bounding with good gradient properties
        activated = torch.sigmoid(x)
        
        # Simple normalization
        return activated / (torch.sum(activated, dim=-1, keepdim=True) + 1e-10)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        # Get probabilities
        probs = self.forward(logits)
        
        # Convert targets to one-hot
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        
        # Simple cross entropy with minimal overhead
        loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-10), dim=-1)
        
        # Handle ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            loss = loss[mask]
        
        return loss.mean()

class SoftmaxActivation(OutputActivation):
    """Standard softmax activation with cross entropy loss for comparison."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1),
            ignore_index=ignore_index
        )

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    activation: ActivationConfig = field(default_factory=get_default_activation_config)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight 

        # Create activation function
        self.output_activation = config.activation.get_activation()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = self.output_activation.loss(logits.view(-1, logits.size(-1)), 
                                             targets.view(-1),
                                             ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply activation to get probabilities
            probs = self.output_activation(logits)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
