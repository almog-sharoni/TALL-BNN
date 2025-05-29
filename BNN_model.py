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
        """
        feat = self.backbone.features(x)           # (B, D) before last FC
        B, D = feat.shape
        C = self.backbone.fc_out.out_features
        votes = torch.zeros(B, C, device=feat.device)

        # Pre-binarise the clean feature once for speed
        feat = binarize(feat)

        for _ in range(self.num_iter):
            # Bernoulli mask: 1 where we flip the sign
            mask = torch.bernoulli(torch.full_like(feat, self.flip_p))
            flipped = feat * (1.0 - 2.0 * mask)    # sign flip: x↦−x where mask==1
            logits = self.backbone.fc_out(flipped) # still ±1 after binarisation
            logits = binarize(logits)              # ensure sign (eval mode)
            votes += (logits + 1) / 2              # map {−1, +1} → {0, 1}

        return votes.argmax(dim=-1)                # final prediction

# ---------- helper factory functions -------------------------------------------
def build_cam4_deep(num_classes: int = 10) -> BinaryMLP:
    return BinaryMLP(hidden_sizes=(4096, 4096, 128), num_classes=num_classes)

def build_cam4_shallow(num_classes: int = 10) -> BinaryMLP:
    return BinaryMLP(hidden_sizes=(128,), num_classes=num_classes)

# Deployment
# Note: This is a post-training operation, so it should be called after training
# and before inference. It modifies the model to remove BatchNorm layers.
def fold_batch_norm(seq):
    """
    Folds BatchNorm1d into the preceding BinarizeLinear weight & bias.
    Call *after* training, in eval() mode.
    """
    new = []
    i = 0
    while i < len(seq):
        lin = seq[i];  bn = seq[i+1];  act = seq[i+2]
        w, b = lin.weight.data, lin.bias
        gamma, beta = bn.weight.data, bn.bias.data
        mu, var = bn.running_mean, bn.running_var
        std = torch.sqrt(var + bn.eps)

        w.mul_(gamma.view(-1, 1) / std.view(-1, 1))
        if b is None:
            b = -mu * gamma / std + beta
        else:
            b = (b - mu) * gamma / std + beta
        lin.bias = nn.Parameter(b)

        new += [lin, act]              # BN removed, bias embedded
        i += 3
    return nn.Sequential(*new)

def clamp_bn_constants(lin: BinarizeLinear,
                       C_max: int = 32,      # 16 / 32 / 64 / 128 …
                       even_only: bool = True):
    """
    Turn the FP32 bias produced by `fold_batch_norm` into the
    integer constant that will be realised with up to C_max
    extra ±1 cells in hardware.
    """
    if lin.bias is None:
        raise ValueError("fold_batch_norm must run first")

    # 1. round to nearest integer
    C = torch.round(lin.bias.data)

    # 2. optionally snap to even numbers
    if even_only:
        C = 2 * torch.round(C / 2)

    # 3. hard-clip to ±C_max
    C.clamp_(-C_max, C_max)

    # 4. store back – from this point on the bias *is* the limited BN constant
    lin.bias.data = C
    lin.bias.requires_grad_(False)      # freeze

# ---------- quick sanity check --------------------------------------------------
if __name__ == "__main__":
    net = build_cam4_deep()
    tall = TALLClassifier(net, num_iter=5, flip_p=0.3).eval()
    dummy = torch.randn(4, 1, 28, 28)            # fake MNIST batch
    print("TALL predictions:", tall(dummy))      # → tensor([…, …, …, …])