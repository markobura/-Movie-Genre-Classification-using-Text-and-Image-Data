"""Dataset that serves the poster and the plot of the same movie."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

from helpers import clean_plot
from poster.config import GENRES
from poster.transforms import get_transforms

PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024


class MultimodalMovieDataset(Dataset):
    def __init__(
        self,
        split_dir: Path | str,
        tokenizer,
        *,
        max_length: int = 256,
        train: bool = False,
    ):
        self.split_dir = Path(split_dir)
        split = self.split_dir.name.replace("_data", "")

        self.labels_df = pd.read_csv(self.split_dir / f"{split}_labels.csv")
        self.image_dir = self.split_dir / "images"
        self.movie_ids = self.labels_df["movie_id"].astype(int).tolist()
        self.targets = self.labels_df[GENRES].values.astype("float32")

        self.transform = get_transforms("clip_vit_b32", train=train)

        plots = clean_plot(self.labels_df["plot"]).tolist()
        encoded = tokenizer(
            plots,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.movie_ids)

    def __getitem__(self, idx: int) -> dict:
        movie_id = self.movie_ids[idx]
        image = Image.open(self.image_dir / f"{movie_id}.jpg").convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.targets[idx]),
            "movie_id": movie_id,
        }


def collate_multimodal(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Pad token sequences to the longest plot in the batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = item["attention_mask"]

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack([item["labels"] for item in batch]),
        "movie_id": torch.tensor([item["movie_id"] for item in batch], dtype=torch.long),
    }
