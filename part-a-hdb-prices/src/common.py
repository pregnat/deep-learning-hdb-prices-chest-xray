"""Shared helpers for questions A1-A4: seeding, training with history, metrics, plots."""

import copy
import dataclasses
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import prepare_data
from model import PriceModel

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = ROOT / "outputs"

# Seeds used for every multi-seed experiment (A2, A3). Seed 42 is also the A1 seed.
SEEDS = (42, 43, 44, 45, 46)

# Colour-blind-safe categorical palette (fixed order).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def combined_config(config):
    """Train on 2017-2021 (train + validation years) and test on 2022."""
    years = config.train_years + (config.validation_year,)
    return dataclasses.replace(config, train_years=years)


def build_model(data, config):
    return PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, dataset, batch_size=4096):
    model.eval()
    predictions, targets = [], []
    for categorical, continuous, target in DataLoader(dataset, batch_size=batch_size):
        predictions.append(model.predict_price(categorical, continuous))
        targets.append(target)
    return torch.cat(predictions), torch.cat(targets)


def evaluate(model, dataset):
    """Return RMSE (SGD), R² and mean bias (prediction - target, SGD) on a dataset."""
    prediction, target = predict(model, dataset)
    squared_error = (prediction - target).square().sum()
    total = (target - target.mean()).square().sum()
    rmse = (squared_error / len(target)).sqrt().item()
    bias = (prediction - target).mean().item()
    return {"rmse": rmse, "r2": 1 - (squared_error / total).item(), "bias": bias}


def train_model(model, train_data, eval_data, config, fixed_epochs=None, on_epoch=None):
    """Train with Adam.

    fixed_epochs=None: early stopping on eval_data (patience/threshold from config),
        restoring the best weights. Used when eval_data is the validation set.
    fixed_epochs=n: train exactly n epochs (the best epoch found with early stopping
        on validation). Used when retraining on train+validation; eval_data is the
        test set and is only *monitored* for the learning-curve plot, never used
        for any decision.

    on_epoch(epoch, metrics) is called after every epoch (used to report to Ray Tune).
    Returns a history dict with per-epoch train/eval RMSE and R², and best_epoch.
    """
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"train_rmse": [], "eval_rmse": [], "train_r2": [], "eval_r2": []}

    best_rmse = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    max_epochs = fixed_epochs or config.max_epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()
            standard_target = (target - model.target_mean) / model.target_std
            loss = loss_function(model(categorical, continuous), standard_target)
            loss.backward()
            optimiser.step()

        # Train metrics are measured at the end of the epoch (eval mode) so they are
        # directly comparable with the eval-set metrics on the same axes.
        train_metrics = evaluate(model, train_data)
        eval_metrics = evaluate(model, eval_data)
        history["train_rmse"].append(train_metrics["rmse"])
        history["eval_rmse"].append(eval_metrics["rmse"])
        history["train_r2"].append(train_metrics["r2"])
        history["eval_r2"].append(eval_metrics["r2"])
        if on_epoch:
            on_epoch(epoch, {"train_rmse": train_metrics["rmse"],
                             "eval_rmse": eval_metrics["rmse"]})

        if fixed_epochs:
            best_epoch = epoch
            continue
        if eval_metrics["rmse"] < best_rmse - config.early_stopping_threshold:
            best_rmse = eval_metrics["rmse"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    history["best_epoch"] = best_epoch
    return history


def run_final(config, seed, epochs, data=None):
    """Retrain on train+validation for a fixed number of epochs and test on 2022."""
    config = combined_config(config)
    data = data or prepare_data(DATA_PATH, config)
    set_seed(seed)
    model = build_model(data, config)
    history = train_model(model, data["train"], data["test"], config, epochs)
    return model, history, evaluate(model, data["test"])


def load_selected():
    """Config and epoch count selected in A1 (written by a1_tune.py)."""
    summary = json.loads((OUTPUT_DIR / "a1" / "summary.json").read_text())
    selected = summary["selected_config"]
    config = dataclasses.replace(
        Config(),
        hidden_width=selected["hidden_width"],
        embedding_dims=tuple(selected["embedding_dims"]),
    )
    return config, selected["epochs"]


def summarise(values):
    """Mean and sample standard deviation (ddof=1)."""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "values": [float(v) for v in values]}


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


def thousands(ax):
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k")
    )


def plot_curves(curves, ylabel, path, title=None, money=True):
    """curves: list of (label, values). One line per curve on shared axes."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for colour, (label, values) in zip(COLOURS, curves):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, color=colour, linewidth=2, marker="o", markersize=4,
                label=label)
    style_axes(ax, "Epoch", ylabel, title)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if money:
        thousands(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
