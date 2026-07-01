import math
import torch
import torch.nn as nn

# Simplified AdaLoRA variant with learnable importance weights for each low-rank component
class AdaLoRALinear(nn.Module):

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
        # Unlike the original AdaLoRA, importance scores are learned directly
        # instead of being updated through dynamic rank reallocation
        # Learnable importance score assigned to each LoRA rank
        self.rank_importance = nn.Parameter(torch.ones(rank))
 
    # Apply the importance-weighted LoRA adaptation
    def forward(
        self,
        x: torch.Tensor
    ):
        # Frozen backbone output
        base = self.linear(x)
        # Weight each low-rank component according to its learned importance
        A = self.A * self.rank_importance[:, None]
        # Compute the importance-weighted LoRA residual
        lora = (x @ A.t()) @ self.B.t()
        # Final adapted projection
        return base + self.scaling * lora