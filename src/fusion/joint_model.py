"""CLIP image encoder + MiniLM text encoder + LateFusionHead in one graph.

Trainable by default: the last CLIP block with post_layernorm and
visual_projection, the last two MiniLM blocks, and the fusion head. The rest
stays frozen and in eval mode.
"""

import torch
import torch.nn as nn

from poster.config import GENRES

from .late_fusion import LateFusionHead

VISION_MODEL_ID = "openai/clip-vit-base-patch32"
TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean of the token vectors, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class JointFusionModel(nn.Module):
    def __init__(
        self,
        n_classes: int = len(GENRES),
        *,
        unfreeze_vision: int = 1,
        unfreeze_text: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        from transformers import AutoModel, CLIPVisionModelWithProjection

        self.vision = CLIPVisionModelWithProjection.from_pretrained(VISION_MODEL_ID)
        self.text = AutoModel.from_pretrained(TEXT_MODEL_ID)

        self.image_dim = self.vision.config.projection_dim
        self.text_dim = self.text.config.hidden_size
        self.unfreeze_vision = unfreeze_vision
        self.unfreeze_text = unfreeze_text

        self.head = LateFusionHead(
            [self.image_dim, self.text_dim],
            n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            normalize_inputs=True,
        )

        self._trainable_encoder_modules = []
        self._freeze_encoders()

    def _freeze_encoders(self) -> None:
        for param in self.vision.parameters():
            param.requires_grad = False
        for param in self.text.parameters():
            param.requires_grad = False

        modules = []
        if self.unfreeze_vision > 0:
            layers = self.vision.vision_model.encoder.layers
            modules.extend(layers[-self.unfreeze_vision:])
            modules.append(self.vision.vision_model.post_layernorm)
            modules.append(self.vision.visual_projection)
        if self.unfreeze_text > 0:
            modules.extend(self.text.encoder.layer[-self.unfreeze_text:])

        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

        self._trainable_encoder_modules = modules

    def train(self, mode: bool = True):
        """Training mode for the unfrozen blocks only; frozen ones stay in eval
        so their dropout does not fire."""
        super().train(mode)
        if mode:
            self.vision.eval()
            self.text.eval()
            for module in self._trainable_encoder_modules:
                module.train()
        return self

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float):
        """Pretrained blocks and the new head get different learning rates."""
        encoder_params = [p for m in self._trainable_encoder_modules for p in m.parameters()]
        groups = [{"params": self.head.parameters(), "lr": head_lr, "weight_decay": weight_decay}]
        if encoder_params:
            groups.append(
                {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}
            )
        return groups

    def trainable_state_dict(self) -> dict:
        """Unfrozen parameters only, so checkpoints stay small."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable}

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "head": sum(p.numel() for p in self.head.parameters()),
        }

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=pixel_values).image_embeds

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return mean_pool(hidden, attention_mask)

    def forward(self, pixel_values, input_ids, attention_mask) -> torch.Tensor:
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        return self.head(image_features, text_features)
