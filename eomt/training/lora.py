import math
import torch
import torch.nn as nn

# Standard LoRA module that augments a frozen linear layer
# with a trainable low-rank residual update
class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
    ):
        super().__init__()
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
 
    # Compute the frozen linear projection plus the LoRA residual
    def forward(
        self,
        x: torch.Tensor
    ):
        # Frozen backbone output
        base = self.linear(x)
        # Low-rank adaptation scaled according to the LoRA formulation
        lora = (x @ self.A.t()) @ self.B.t()

        # Final adapted projection
        return base + self.scaling * lora