# cam4_bnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# ---------- low-level primitives ------------------------------------------------
class SignSTE(Function):
    """sign(x) with straight-through gradient"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        grad = g.clone()
        grad[x.abs() > 1] = 0.0         # clamp outside (-1,1)
        return grad


def binarize(x):
    """Deterministic binarisation (+1 / –1)"""
    return x.sign()


class BinaryActivation(nn.Module):
    """Hard sign with STE"""
    def forward(self, x):
        if self.training:
            return SignSTE.apply(x)
        return binarize(x)


class BinarizeLinear(nn.Linear):
    """
    Fully-connected layer whose inputs and weights are both binarised.
    During the forward pass we copy self.weight so that PyTorch autograd
    still owns an FP32 tensor to update.
    """
    def forward(self, x):
        if self.training:
            w_bin = SignSTE.apply(self.weight)
            x_bin = SignSTE.apply(x)
        else:
            w_bin = binarize(self.weight)
            x_bin = binarize(x)
        return F.linear(x_bin, w_bin, self.bias)

# ---------- backbone network ----------------------------------------------------
class BinaryMLP(nn.Module):
    """
    Default is the *deep* 784→4096→4096→128→10 model from Hubara (2016).
    Switch hidden_sizes to (128,) for the *shallow* variant used in Jung (2022).
    """
    def __init__(self,
                 in_features: int = 28 * 28,
                 hidden_sizes: tuple[int, ...] = (4096, 4096, 128),
                 num_classes: int = 10):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers += [
                BinarizeLinear(prev, h, bias=False),
                nn.BatchNorm1d(h),            # FP32 during training
                BinaryActivation()
            ]
            prev = h

        self.hidden = nn.Sequential(*layers)
        self.fc_out = BinarizeLinear(prev, num_classes, bias=False)

    # forward that returns raw logits (sign outputs, ±1) – useful for TALL
    def forward(self, x):
        x = x.flatten(1)                    # (B, 784)
        x = self.hidden(x)
        return self.fc_out(x)

    # expose last-hidden activations so TALL can reuse them
    def features(self, x):
        x = x.flatten(1)
        return self.hidden(x)

# ---------- Time-Augmented Last Layer (TALL) voting -----------------------------
class TALLClassifier(nn.Module):
    """
    Wraps a BinaryMLP and replaces the softmax with majority voting over
    `num_iter` noisy passes whose bits are flipped with probability `flip_p`.
    """
    def __init__(self,
                 backbone: BinaryMLP,
                 num_iter: int = 30,
                 flip_p: float = 0.30):
        super().__init__()
        assert 0.0 <= flip_p <= 1.0
        self.backbone = backbone
        self.num_iter = num_iter
        self.flip_p = flip_p

    @torch.no_grad()
    def forward(self, x):
        """
        Returns predicted class indices (shape: [batch]).  
        Majority is implemented by counting +1s for each class across iterations.
        
        Optimized version using boolean operations instead of float arithmetic.
        """
        feat = self.backbone.features(x)           # (B, D) before last FC
        B, D = feat.shape
        C = self.backbone.fc_out.out_features
        votes = torch.zeros(B, C, device=feat.device, dtype=torch.long)

        # Pre-binarise the clean feature once for speed
        feat = binarize(feat)
        feat_sign = feat > 0  # Convert to boolean mask for efficient operations

        for _ in range(self.num_iter):
            # Boolean mask: True where we flip the sign
            flip_mask = torch.rand_like(feat, dtype=torch.float32) < self.flip_p
            
            # Apply flips: XOR with flip mask (True flips sign, False keeps it)
            flipped_sign = feat_sign ^ flip_mask
            
            # Convert back to {-1, +1} for linear layer
            flipped = flipped_sign.float() * 2.0 - 1.0
            
            logits = self.backbone.fc_out(flipped) # still ±1 after binarisation
            logits = binarize(logits)              # ensure sign (eval mode)
            
            # Count positive votes (logits > 0) - more efficient than float conversion
            positive_votes = (logits > 0).long()
            votes += positive_votes

        return votes.argmax(dim=-1)                # final prediction

# ---------- helper factory functions -------------------------------------------
def build_cam4_deep(num_classes: int = 10) -> BinaryMLP:
    return BinaryMLP(hidden_sizes=(4096, 4096, 128), num_classes=num_classes)

def build_cam4_shallow(num_classes: int = 10) -> BinaryMLP:
    return BinaryMLP(hidden_sizes=(128,), num_classes=num_classes)


# ---------- quick sanity check --------------------------------------------------
if __name__ == "__main__":
    net = build_cam4_deep()
    tall = TALLClassifier(net, num_iter=5, flip_p=0.3).eval()
    dummy = torch.randn(4, 1, 28, 28)            # fake MNIST batch
    print("TALL predictions:", tall(dummy))      # → tensor([…, …, …, …])

# ---------- Post-training optimization functions --------------------------------

class BinarizeLinearWithFoldedBN(nn.Module):
    """
    Hybrid BNN BatchNorm folding for hardware deployment.
    
    This layer preserves exact BatchNorm computation in FP32 mode and provides
    hardware-compatible integer constants in hardware mode. This avoids the
    fundamental issue that standard BN folding breaks weight binarization.
    
    Modes:
    1. FP32 mode: Exact BatchNorm computation (perfect accuracy)
    2. Hardware mode: Integer bias constants C_j ∈ [-c_max, +c_max]
    """
    
    def __init__(self, in_features, out_features, weight, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Binary weights (unchanged from original BinarizeLinear)
        self.weight = nn.Parameter(weight)
        
        # FP32 BatchNorm parameters (preserved for exact computation)
        self.register_buffer('bn_gamma', bn_gamma)
        self.register_buffer('bn_beta', bn_beta)
        self.register_buffer('bn_mean', bn_mean)
        self.register_buffer('bn_var', bn_var)
        self.bn_eps = bn_eps
        
        # Hardware deployment mode
        self.hardware_mode = False
        self.bias_constants = None
        
    def enable_hardware_mode(self, c_max=32, even_only=True):
        """
        Enable hardware mode with quantized integer constants.
        
        Args:
            c_max: Maximum absolute value for bias constants
            even_only: If True, clamp to even integers only for hardware efficiency
        """
        with torch.no_grad():
            # Compute correct FP32 bias: C_j = β_j * sqrt(σ_j² + ε) / γ_j - μ_j
            std = torch.sqrt(self.bn_var + self.bn_eps)
            fp32_bias = self.bn_beta * std / self.bn_gamma - self.bn_mean
            
            # Handle negative gamma values by flipping weight signs
            # This keeps binary weights truly ±1 without folding errors
            neg_gamma_mask = (self.bn_gamma < 0)
            if neg_gamma_mask.any():
                # Flip weights for channels with negative gamma
                neg_mask = neg_gamma_mask.view(-1, 1)
                self.weight.data[neg_mask.squeeze(), :] *= -1
                # Also flip the gamma and bias accordingly
                self.bn_gamma.data[neg_gamma_mask] *= -1
                fp32_bias[neg_gamma_mask] *= -1
            
            # Quantize to integers in range [-c_max, +c_max]
            quantized = torch.clamp(torch.round(fp32_bias), -c_max, c_max)
            
            # Force even integers for hardware efficiency
            if even_only:
                quantized = torch.round(quantized / 2) * 2
                quantized = torch.clamp(quantized, -c_max, c_max)
            
            self.bias_constants = quantized
            
            # Register as parameter for proper export
            if hasattr(self, 'bias') and self.bias is not None:
                # Remove existing bias parameter
                del self.bias
            self.register_parameter('bias', nn.Parameter(self.bias_constants.clone(), requires_grad=False))
            self.hardware_mode = True
            
        return self.bias_constants
        
    def disable_hardware_mode(self):
        """Disable hardware mode and return to FP32 BatchNorm"""
        self.hardware_mode = False
        self.bias_constants = None
        # Remove the bias parameter
        if hasattr(self, 'bias') and self.bias is not None:
            delattr(self, 'bias')
            self.bias = None
        
    def forward(self, x):
        # Step 1: Binarize input and weights (same as BinarizeLinear)
        if self.training:
            x_bin = SignSTE.apply(x)
            w_bin = SignSTE.apply(self.weight)
        else:
            x_bin = binarize(x)
            w_bin = binarize(self.weight)
        
        # Step 2: Linear transformation (no bias)
        out = F.linear(x_bin, w_bin, None)
        
        # Step 3: Apply normalization/bias
        if self.hardware_mode and hasattr(self, 'bias') and self.bias is not None:
            # Hardware mode: add quantized integer constants
            out = out + self.bias
        else:
            # FP32 mode: exact BatchNorm computation
            std = torch.sqrt(self.bn_var + self.bn_eps)
            out = (out - self.bn_mean) / std * self.bn_gamma + self.bn_beta
            
        return out


def fold_batch_norm(seq):
    """
    Fold BatchNorm1d layers into preceding BinarizeLinear layers for hardware deployment.
    
    This function processes sequences of [BinarizeLinear, BatchNorm1d, BinaryActivation]
    and creates BinarizeLinearWithFoldedBN layers that preserve the exact computation.
    
    Args:
        seq: Sequential module containing triplets of (BinarizeLinear, BatchNorm1d, BinaryActivation)
        
    Returns:
        New Sequential module with BatchNorm layers folded into linear layers
    """
    new = []
    i = 0
    
    while i < len(seq):
        # Check if we have enough layers for a triplet
        if i + 2 >= len(seq):
            # Not enough layers for a triplet, just copy remaining
            new.extend(seq[i:])
            break
            
        lin = seq[i]
        bn = seq[i+1]
        act = seq[i+2]
        
        # Verify we have the expected layer types
        if not isinstance(lin, BinarizeLinear):
            # Not a BinarizeLinear, just copy and continue
            new.append(lin)
            i += 1
            continue
            
        if not isinstance(bn, nn.BatchNorm1d):
            # Not a BatchNorm1d, copy both and continue
            new.extend([lin, bn])
            i += 2
            continue
        
        # Create a folded layer that combines BinarizeLinear + BatchNorm
        # Use hybrid approach: preserve exact BN computation with optional hardware mode
        folded_layer = BinarizeLinearWithFoldedBN(
            lin.in_features,
            lin.out_features,
            lin.weight.data.clone(),  # Keep original binary weights
            bn.weight.data.clone(),
            bn.bias.data.clone(), 
            bn.running_mean.clone(),
            bn.running_var.clone(),
            bn.eps
        )
        
        # Add folded layer + activation (skip original linear and BatchNorm)
        new.extend([folded_layer, act])
        i += 3
    
    return nn.Sequential(*new)


def clamp_bn_constants(layer, c_max: int = 32, even_only: bool = True):
    """
    Enable hardware mode with quantized integer bias constants.
    
    This function switches a BinarizeLinearWithFoldedBN layer from FP32 mode
    (exact BatchNorm) to hardware mode (integer constants C_j).
    
    Args:
        layer: BinarizeLinearWithFoldedBN layer
        c_max: Maximum absolute value for bias constants (16/32/64/128...)
        even_only: If True, snap to even integers only (hardware constraint)
    
    Returns:
        Quantization statistics for analysis
    """
    if not isinstance(layer, BinarizeLinearWithFoldedBN):
        raise ValueError("Layer must be BinarizeLinearWithFoldedBN")

    # Enable hardware mode and get the bias constants
    bias_constants = layer.enable_hardware_mode(c_max, even_only)
    
    # Compute FP32 reference for comparison
    with torch.no_grad():
        std = torch.sqrt(layer.bn_var + layer.bn_eps)
        fp32_bias = layer.bn_beta * std / layer.bn_gamma - layer.bn_mean
        
        # Quantization statistics
        quant_error = torch.abs(bias_constants - fp32_bias)
        stats = {
            'max_error': quant_error.max().item(),
            'mean_error': quant_error.mean().item(),
            'num_clipped': (torch.abs(fp32_bias) > c_max).sum().item(),
            'num_neurons': len(fp32_bias),
            'fp32_range': (fp32_bias.min().item(), fp32_bias.max().item()),
            'quantized_range': (bias_constants.min().item(), bias_constants.max().item())
        }
    
    return stats


def apply_post_training_optimization(model, c_max: int = 32, even_only: bool = True):
    """
    Apply complete post-training optimization pipeline to a BNN model.
    
    This function:
    1. Folds BatchNorm into preceding linear layers
    2. Clamps the resulting bias constants for hardware compatibility
    
    Args:
        model: Trained BNN model (should be in eval mode)
        c_max: Maximum absolute value for bias constants
        even_only: Whether to use only even integer constants (default: True for hardware efficiency)
        
    Returns:
        Optimized model ready for hardware deployment
    """
    if not isinstance(model, BinaryMLP):
        raise ValueError("Model must be a BinaryMLP instance")
    
    model.eval()  # Ensure we're in eval mode
    
    # Apply batch norm folding to hidden layers
    model.hidden = fold_batch_norm(model.hidden)
    
    # Apply bias clamping to all folded layers
    for module in model.modules():
        if isinstance(module, BinarizeLinearWithFoldedBN):
            clamp_bn_constants(module, c_max, even_only)
    
    return model


def export_hardware_weights(model, output_path: str):
    """
    Export model weights in hardware-compatible format.
    
    Args:
        model: Post-training optimized BNN model
        output_path: Path to save the weights file (.npz format)
    """
    import numpy as np
    
    weights_dict = {}
    layer_idx = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (BinarizeLinear, BinarizeLinearWithFoldedBN)):
            # Binarize weights to {-1, +1} then convert to bits
            binary_weights = torch.sign(module.weight.data).cpu().numpy()
            
            # Convert {-1, +1} to {0, 1} for bit packing
            bit_weights = ((binary_weights + 1) / 2).astype(np.uint8)
            
            weights_dict[f'layer_{layer_idx}_weights'] = bit_weights
            
            # Export bias constants if present
            if hasattr(module, 'bias') and module.bias is not None:
                bias_constants = module.bias.data.cpu().numpy().astype(np.int8)
                weights_dict[f'layer_{layer_idx}_bias'] = bias_constants
            
            layer_idx += 1
    
    np.savez_compressed(output_path, **weights_dict)
    print(f"Hardware weights exported to: {output_path}")


def create_hardware_checkpoint(model, original_checkpoint_path: str, output_path: str, c_max: int):
    """
    Create a hardware-ready checkpoint with all necessary metadata.
    
    Args:
        model: Post-training optimized model
        original_checkpoint_path: Path to original training checkpoint
        output_path: Path to save hardware checkpoint
        c_max: C_MAX value used for bias clamping
    """
    # Load original checkpoint for metadata
    original_checkpoint = torch.load(original_checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create hardware checkpoint
    hardware_checkpoint = {
        'model_state_dict': model.state_dict(),
        'hardware_ready': True,
        'c_max': c_max,
        'bn_folded': True,
        'fully_binarized': True,
        'original_epoch': original_checkpoint.get('epoch', 'unknown'),
        'original_accuracy': original_checkpoint.get('best_acc', 'unknown'),
        'export_timestamp': torch.tensor(0),  # Current timestamp would go here
        'model_architecture': 'BinaryMLP',
        'layer_count': sum(1 for _ in model.modules() if isinstance(_, (BinarizeLinear, BinarizeLinearWithFoldedBN))),
    }
    
    torch.save(hardware_checkpoint, output_path)
    print(f"Hardware checkpoint saved to: {output_path}")