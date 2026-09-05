from pathlib import Path

import pandas as pd
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

from .config import GENRES

# PNG posters with large metadata exceed PIL's default chunk limit
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024


class PosterDataset(Dataset):
    def __init__(self, split_dir: Path | str, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform

        split_name = self.split_dir.name.replace("_data", "")
        labels_path = self.split_dir / f"{split_name}_labels.csv"

        self.labels_df = pd.read_csv(labels_path)
        self.image_dir = self.split_dir / "images"
        self.movie_ids = self.labels_df["movie_id"].astype(int).tolist()
        self.targets = self.labels_df[GENRES].values.astype("float32")

    def __len__(self) -> int:
        return len(self.movie_ids)

    def __getitem__(self, idx: int):
        movie_id = self.movie_ids[idx]
        image = Image.open(self.image_dir / f"{movie_id}.jpg").convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.targets[idx], movie_id
