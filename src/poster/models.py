import torch.nn as nn
from torchvision import models

from .config import GENRES


def load_vgg16_imagenet():
    weights = models.VGG16_Weights.IMAGENET1K_V1
    return models.vgg16(weights=weights)


def freeze_vgg_features(model):
    for param in model.features.parameters():
        param.requires_grad = False
    return model


def replace_vgg_classifier(model, num_classes=len(GENRES), dropout=0.0):
    in_features = model.classifier[0].in_features
    layers = [
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.extend([nn.Linear(4096, 1000), nn.ReLU(True)])
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(1000, num_classes))
    model.classifier = nn.Sequential(*layers)
    return model


def build_vgg16_stanford(num_classes=len(GENRES)):
    model = load_vgg16_imagenet()
    freeze_vgg_features(model)
    replace_vgg_classifier(model, num_classes, dropout=0.0)
    return model


def build_vgg16_regularized(num_classes=len(GENRES), dropout=0.5):
    model = load_vgg16_imagenet()
    freeze_vgg_features(model)
    replace_vgg_classifier(model, num_classes, dropout=dropout)
    return model


def build_clip_vit_b32(num_classes=len(GENRES)):
    from transformers import CLIPModel

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    for param in clip.vision_model.parameters():
        param.requires_grad = False

    embed_dim = clip.config.projection_dim

    class ClipGenreHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = clip
            self.head = nn.Linear(embed_dim, num_classes)

        def forward(self, pixel_values):
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            if pooled is None:
                pooled = vision_outputs.last_hidden_state[:, 0, :]
            image_embeds = self.clip.visual_projection(pooled)
            return self.head(image_embeds)

    return ClipGenreHead()


def build_model(backbone: str, num_classes=len(GENRES)):
    if backbone == "vgg16_stanford":
        return build_vgg16_stanford(num_classes)
    if backbone == "vgg16_regularized":
        return build_vgg16_regularized(num_classes)
    if backbone == "clip_vit_b32":
        return build_clip_vit_b32(num_classes)
    raise ValueError(f"Unknown backbone: {backbone}")
