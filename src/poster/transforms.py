from torchvision import transforms


def to_tensor_transform():
    return transforms.ToTensor()


def resize_center_crop_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
