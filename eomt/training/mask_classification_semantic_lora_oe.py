import torch

from training.mask_classification_semantic_oe import MaskClassificationSemanticOE

from training.lora import LoRALinear
from training.selective_lora import SelectiveLoRALinear
from training.ada_lora import AdaLoRALinear

class MaskClassificationSemanticLoRAOE(MaskClassificationSemanticOE):
    def __init__(
        self,
        lora_mode="standard",
        lora_rank=8,
        lora_alpha=16,
        target_blocks=(9,10,11),
        **kwargs
    ):

        super().__init__(**kwargs)

        self.lora_mode = lora_mode
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_blocks = target_blocks
        self.inject_lora()
        self.freeze_except_lora()
        self.print_trainable_parameters()

    def get_lora_class(self):

        if self.lora_mode == "standard":
            return LoRALinear

        elif self.lora_mode == "selective":
            return SelectiveLoRALinear

        elif self.lora_mode == "adalora":
            return AdaLoRALinear

        raise ValueError(f"Unknown lora_mode: {self.lora_mode}")

    def inject_lora(self):

        LoRAClass = self.get_lora_class()
        num_injected = 0

        blocks = self.network.encoder.backbone.blocks

        for idx, block in enumerate(blocks):
            if idx not in self.target_blocks:
                continue

            block.attn.qkv = LoRAClass(
                block.attn.qkv,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
            )

            block.attn.proj = LoRAClass(
                block.attn.proj,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
            )

            num_injected += 2

            print(f"LoRA injected into block {idx}")
        print(f"Injected LoRA into {num_injected} layers")

    def freeze_except_lora(self):
        for param in self.network.parameters():
            param.requires_grad = False
        """
        for module in self.network.modules():
            if isinstance(module, (LoRALinear, SelectiveLoRALinear, AdaLoRALinear)):
                for param in module.parameters():
                    param.requires_grad = True
        """
 
        for module in self.network.modules():
            if isinstance(module, LoRALinear):
                module.A.requires_grad = True
                module.B.requires_grad = True
 
            if isinstance(module, SelectiveLoRALinear): 
                module.A.requires_grad = True
                module.B.requires_grad = True
                module.gate.requires_grad = True
 
            if isinstance(module, AdaLoRALinear):
                module.A.requires_grad = True
                module.B.requires_grad = True
                module.rank_importance.requires_grad = True
 
 
    def print_trainable_parameters(self):
        trainable = 0
        total = 0

        for param in self.network.parameters():
            n = param.numel()
            total += n

            if param.requires_grad:
                trainable += n

        percent = 100.0 * trainable / total

        print("\nLoRA")
        print(f"Mode          : {self.lora_mode}")
        print(f"Rank          : {self.lora_rank}")
        print(f"Alpha         : {self.lora_alpha}")
        print(f"Target blocks : {self.target_blocks}")
        print(f"Trainable     : {trainable:,}")
        print(f"Total         : {total:,}")
        print(f"Percentage    : {percent:.4f}%")  