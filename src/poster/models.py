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


def replace_vgg_classifier(model, num_classes=len(GENRES)):
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Linear(4096, 1000),
        nn.ReLU(True),
        nn.Linear(1000, num_classes),
    )
    return model

