from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from poster.metrics import compute_ap_metrics


class LateFusionHead(nn.Module):
    """Concatenate per-modality vectors -> genre logits.

    in_dims          width of each modality, [13, 13] for probabilities or
                     [512, 384] for CLIP + MiniLM embeddings
    hidden_dim       0 keeps a single linear layer, >0 adds one ReLU layer
    normalize_inputs LayerNorm per modality before concatenation
    """

    def __init__(
        self,
        in_dims: Sequence[int],
        n_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        normalize_inputs: bool = False,
    ):
        super().__init__()
        self.in_dims = list(in_dims)
        self.norms = (
            nn.ModuleList([nn.LayerNorm(d) for d in self.in_dims])
            if normalize_inputs
            else None
        )

        in_features = sum(self.in_dims)
        if hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.fc = nn.Linear(in_features, n_classes)

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        if len(modalities) != len(self.in_dims):
            raise ValueError(
                f"Expected {len(self.in_dims)} modalities, got {len(modalities)}"
            )
        parts = list(modalities)
        if self.norms is not None:
            parts = [norm(x) for norm, x in zip(self.norms, parts)]
        return self.fc(torch.cat(parts, dim=1))


def _macro_ap(model: LateFusionHead, inputs, y, device) -> float:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(*[x.to(device) for x in inputs])).cpu().numpy()
    return compute_ap_metrics(y.cpu().numpy(), probs)["macro_ap"]


def train_late_fusion(
    train_inputs: Sequence[torch.Tensor],
    y_train: torch.Tensor,
    val_inputs: Sequence[torch.Tensor],
    y_val: torch.Tensor,
    pos_weight: torch.Tensor,
    *,
    hidden_dim: int = 0,
    dropout: float = 0.0,
    normalize_inputs: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 20,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[LateFusionHead, dict]:
    """Fit the head on train, select the checkpoint on val by macro AP.

    train_inputs / val_inputs: one tensor per modality, rows aligned with the
    label matrices.
    """
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    n_classes = y_train.shape[1]
    in_dims = [x.shape[1] for x in train_inputs]
    model = LateFusionHead(
        in_dims,
        n_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        normalize_inputs=normalize_inputs,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_inputs = [x.to(device) for x in train_inputs]
    y_train = y_train.to(device)
    val_inputs = [x.to(device) for x in val_inputs]
    y_val = y_val.to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*train_inputs, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_macro_ap = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            *xs, y = batch
            optimizer.zero_grad()
            loss = criterion(model(*xs), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.shape[0]
        train_loss = total_loss / len(y_train)

        train_macro_ap = _macro_ap(model, train_inputs, y_train, device)
        val_macro_ap = _macro_ap(model, val_inputs, y_val, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_macro_ap": train_macro_ap,
                "val_macro_ap": val_macro_ap,
            }
        )
        if verbose:
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} "
                f"train_macro_ap={train_macro_ap:.4f} val_macro_ap={val_macro_ap:.4f}"
            )

        if val_macro_ap > best_macro_ap:
            best_macro_ap = val_macro_ap
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("Late fusion training did not produce a checkpoint")

    model.load_state_dict(best_state)
    return model, {
        "trained_on": "train",
        "selected_on": "val",
        "best_epoch": best_epoch,
        "best_val_macro_ap": best_macro_ap,
        "train_macro_ap_at_best": history[best_epoch - 1]["train_macro_ap"],
        "history": history,
    }


def predict_fusion(
    model: LateFusionHead,
    inputs: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(*[x.to(device) for x in inputs])
        return torch.sigmoid(logits).cpu()
