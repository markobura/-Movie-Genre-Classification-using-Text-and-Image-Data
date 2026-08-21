from torchvision import models


def load_vgg16_imagenet():
    weights = models.VGG16_Weights.IMAGENET1K_V1
    return models.vgg16(weights=weights)


def freeze_vgg_features(model):
    for param in model.features.parameters():
        param.requires_grad = False
    return model
