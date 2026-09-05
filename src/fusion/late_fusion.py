import torch
import torch.nn as nn

from poster.metrics import compute_ap_metrics


class LateFusionHead(nn.Module):
    """Stanford-style late fusion: concat modality probabilities -> linear head."""

    def __init__(self, n_modalities: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_modalities * n_classes, n_classes)

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        x = torch.cat(modalities, dim=1)
        return self.fc(x)


def train_late_fusion(
    poster_val: torch.Tensor,
    text_val: torch.Tensor,
    y_val: torch.Tensor,
    pos_weight: torch.Tensor,
    *,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[LateFusionHead, dict]:
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    model = LateFusionHead(n_modalities=2, n_classes=y_val.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    poster_val = poster_val.to(device)
    text_val = text_val.to(device)
    y_val = y_val.to(device)

    best_macro_ap = 0.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(poster_val, text_val)
        loss = criterion(logits, y_val)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
        metrics = compute_ap_metrics(y_val.cpu().numpy(), probs)
        val_macro_ap = metrics["macro_ap"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_macro_ap": val_macro_ap,
            }
        )

        if val_macro_ap > best_macro_ap:
            best_macro_ap = val_macro_ap
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Late fusion training did not produce a checkpoint")

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_ap": best_macro_ap,
        "history": history,
    }


def predict_fusion(
    model: LateFusionHead,
    poster: torch.Tensor,
    text: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(poster.to(device), text.to(device))
        return torch.sigmoid(logits).cpu()
