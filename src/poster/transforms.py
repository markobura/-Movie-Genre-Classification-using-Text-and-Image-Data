from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


def to_tensor_transform():
    return transforms.ToTensor()


def resize_center_crop_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])


def normalize_transform():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def get_transforms(backbone: str, train: bool = True):
    if backbone != "vgg16_stanford":
        raise ValueError(f"Unknown backbone: {backbone}")

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
