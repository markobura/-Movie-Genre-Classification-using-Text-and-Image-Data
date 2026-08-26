import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import build_model


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_checkpoint(checkpoint_path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone = checkpoint.get("backbone", "vgg16_stanford")
    model = build_model(backbone).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def run_inference(model, loader: DataLoader, device: torch.device):
    model.eval()
    y_true_batches = []
    y_score_batches = []
    movie_id_batches = []

    for images, labels, movie_ids in tqdm(loader, desc="infer"):
        images = images.to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(images))

        y_true_batches.append(labels.numpy())
        y_score_batches.append(probs.cpu().numpy())
        movie_id_batches.append(np.asarray(movie_ids))

    y_true = np.concatenate(y_true_batches, axis=0)
    y_score = np.concatenate(y_score_batches, axis=0)
    movie_ids = np.concatenate(movie_id_batches, axis=0)
    return y_true, y_score, movie_ids
