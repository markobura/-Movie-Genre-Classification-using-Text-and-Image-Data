from torchvision import transforms

from .config import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD


def _vgg_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _vgg_train_transform(augment: bool):
    if not augment:
        return _vgg_eval_transform()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _clip_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def _clip_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def get_transforms(backbone: str, train: bool = True):
    if backbone in ("vgg16_stanford", "vgg16_regularized"):
        augment = train and backbone == "vgg16_regularized"
        return _vgg_train_transform(augment=augment)
    if backbone == "clip_vit_b32":
        return _clip_train_transform() if train else _clip_eval_transform()
    raise ValueError(f"Unknown backbone: {backbone}")
