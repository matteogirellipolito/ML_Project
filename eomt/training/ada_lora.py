import math
import torch
import torch.nn as nn


class AdaLoRALinear(nn.Module):

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

        in_features = linear.in_features
        out_features = linear.out_features

        self.A = nn.Parameter(
            torch.zeros(rank, in_features)
        )

        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        # importance dei rank
        self.rank_importance = nn.Parameter(
            torch.ones(rank)
        )

        nn.init.kaiming_uniform_(
            self.A,
            a=math.sqrt(5)
        )

        nn.init.zeros_(self.B)

    def forward(
        self,
        x: torch.Tensor
    ):

        base = self.linear(x)

        A = self.A * self.rank_importance[:, None]

        lora = (x @ A.t()) @ self.B.t()

        return base + self.scaling * lora