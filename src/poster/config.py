from pathlib import Path

GENRES = [
    "action",
    "animation",
    "biography",
    "comedy",
    "crime",
    "drama",
    "family",
    "fantasy",
    "horror",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
]

ROOT = Path(__file__).resolve().parents[2]

# ImageNet stats for VGG backbones (torchvision).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP ViT-B/32 preprocessing (openai/clip-vit-base-patch32).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

BACKBONES = (
    "vgg16_stanford",
    "vgg16_regularized",
    "clip_vit_b32",
)

BACKBONE_WEIGHT_DECAY = {
    "vgg16_stanford": 0.0,
    "vgg16_regularized": 1e-4,
    "clip_vit_b32": 1e-4,
}
