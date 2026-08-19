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

#https://docs.pytorch.org/vision/stable/models.html
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = (
    "vgg16_stanford",
)
