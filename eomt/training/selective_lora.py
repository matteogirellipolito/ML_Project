import math

import torch
import torch.nn as nn

# Simplified Selective LoRA variant with learnable gates controlling
# the contribution of each low-rank component
class SelectiveLoRALinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
    ):
        super().__init__()
        # Frozen linear layer to be adapted
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Extract the dimensions of the original linear layer
        in_features = linear.in_features
        out_features = linear.out_features
 
        # Trainable low-rank decomposition (B @ A)
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
       
        # Follow the original LoRA initialization: random A and zero-initialized B 
        # Kaiming initialization matching PyTorch's default Linear initialization
        # a = sqrt(5) preserves the same variance used by nn.Linear
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        # Learnable gates weighting the contribution of each rank component
        # Unlike standard LoRA, each rank component is weighted independently
        self.gate = nn.Parameter(torch.ones(rank))
 
    # Apply the gated low-rank adaptation on top of the frozen layer
    def forward(
        self,
        x: torch.Tensor
    ):
        # Frozen backbone output
        base = self.linear(x)
        # Scale each low-rank component through its learnable gate
        # Gates are optimized jointly with the LoRA parameters.
        A = self.A * self.gate[:, None]
        # Compute the gated LoRA residual
        lora = (x @ A.t()) @ self.B.t()
        # Final adapted projection
        return base + self.scaling * lora