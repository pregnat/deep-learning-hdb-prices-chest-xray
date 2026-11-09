"""Shared helpers for questions B1-B4: data, training, metrics, model cost, plots."""

import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "chestxray_binary_64.npz"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42

MAX_EPOCHS = 20
PATIENCE = 4
MIN_DELTA = 1e-4

# Colour-blind-safe categorical palette (fixed order).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data():
    """Scale to [0, 1], then standardise with the training-set mean and std."""
    raw = np.load(DATA_PATH)
    train = raw["train_images"].astype(np.float32) / 255.0
    mean, std = float(train.mean()), float(train.std())

    def to_dataset(split):
        images = (raw[f"{split}_images"].astype(np.float32) / 255.0 - mean) / std
        images = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 64, 64)
        labels = torch.from_numpy(raw[f"{split}_labels"]).float()
        return TensorDataset(images, labels)

    return {"train": to_dataset("train"), "validation": to_dataset("val"),
            "test": to_dataset("test"), "mean": mean, "std": std}


@torch.no_grad()
def evaluate(model, dataset, batch_size=512):
    """Loss, accuracy, error, sensitivity (pneumonia recall), specificity (normal recall)."""
    model.eval()
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, predictions, labels = 0.0, [], []
    for images, target in DataLoader(dataset, batch_size=batch_size):
        logits = model(images).squeeze(1)
        total_loss += loss_function(logits, target).item()
        predictions.append((logits > 0).long())
        labels.append(target.long())
    pred, true = torch.cat(predictions), torch.cat(labels)
    tp = ((pred == 1) & (true == 1)).sum().item()
    tn = ((pred == 0) & (true == 0)).sum().item()
    fp = ((pred == 1) & (true == 0)).sum().item()
    fn = ((pred == 0) & (true == 1)).sum().item()
    accuracy = (tp + tn) / len(true)
    return {
        "loss": total_loss / len(true),
        "accuracy": accuracy,
        "error": 1 - accuracy,
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def train_model(model, data, learning_rate, batch_size, label=""):
    """Adam + early stopping on validation loss (patience 4, min delta 1e-4, ≤20 epochs).

    Restores the weights of the best epoch. Returns the per-epoch history and the
    validation metrics at the best epoch.
    """
    set_seed()
    loader = DataLoader(data["train"], batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    best_loss, best_weights, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running, count = 0.0, 0
        for images, target in loader:
            optimiser.zero_grad()
            loss = loss_function(model(images).squeeze(1), target)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(target)
            count += len(target)

        val = evaluate(model, data["validation"])
        history["train_loss"].append(running / count)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        print(f"{label} epoch {epoch:02d} | train loss {running / count:.4f} | "
              f"val loss {val['loss']:.4f} | val acc {val['accuracy']:.4f}")

        if val["loss"] < best_loss - MIN_DELTA:
            best_loss, best_epoch, waited = val["loss"], epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited == PATIENCE:
                break

    model.load_state_dict(best_weights)
    return {"history": history, "best_epoch": best_epoch,
            "validation": evaluate(model, data["validation"])}


def build_and_train(name, data, learning_rate, batch_size):
    """Seeded model construction followed by training (see train_model)."""
    from models import MODELS

    set_seed()
    model = MODELS[name]()
    result = train_model(model, data, learning_rate, batch_size,
                         label=f"{name} lr={learning_rate:g} bs={batch_size}")
    return model, result


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape=(1, 1, 64, 64)):
    """Forward-pass FLOPs for one image (conv + linear layers; 1 MAC = 2 FLOPs)."""
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(input_shape))
    return counter.get_total_flops()


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def style_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#a3a29d")


def plot_validation_curves(runs, key, ylabel, title, path):
    """runs: {label: result from train_model}. Best epoch marked with a ring."""
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for colour, (label, run) in zip(COLOURS, runs.items()):
        values = run["history"][key]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, color=colour, linewidth=2, marker="o", markersize=4, label=label)
        best = run["best_epoch"]
        ax.scatter([best], [values[best - 1]], s=90, facecolors="none", edgecolors=colour,
                   linewidths=1.8, zorder=3)
    style_axes(ax, "Epoch", ylabel, title)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sweep_table(rows, key_name):
    """Print the table required in B1/B2."""
    print(f"\n{key_name:>12} | best epoch | val loss | val error | sensitivity | specificity")
    for r in rows:
        v = r["validation"]
        print(f"{r[key_name]:>12} | {r['best_epoch']:>10} | {v['loss']:.4f}   | "
              f"{v['error']:.4f}    | {v['sensitivity']:.4f}      | {v['specificity']:.4f}")


def select(rows):
    """Lowest validation error; ties broken by lower validation loss."""
    return min(rows, key=lambda r: (round(r["validation"]["error"], 6), r["validation"]["loss"]))
