"""Question A2: ablations of the selected A1 architecture over five seeds.

  baseline        selected A1 model (embeddings + continuous, ReLU)
  no_categorical  (a) embedding tables removed, six continuous features only
  sigmoid         (b) hidden ReLU replaced by Sigmoid

Every run is trained on 2017-2021 for the number of epochs selected in A1 and
evaluated once on 2022. Run from the repository root:  uv run python src/a2_ablation.py
"""

import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from common import (
    COLOURS, DATA_PATH, OUTPUT_DIR, SEEDS, combined_config, count_parameters,
    load_selected, run_final, save_json, style_axes, summarise, thousands,
)
from data import prepare_data

OUT = OUTPUT_DIR / "a2"

LABELS = {
    "baseline": "Baseline (ReLU)",
    "no_categorical": "(a) No embeddings",
    "sigmoid": "(b) Sigmoid",
}


def variants(config):
    return {
        "baseline": config,
        "no_categorical": dataclasses.replace(config, use_categorical=False),
        "sigmoid": dataclasses.replace(config, activation="sigmoid"),
    }


def run_seeds(name, config, epochs, data):
    runs = []
    for seed in SEEDS:
        model, history, test = run_final(config, seed, epochs, data)
        print(f"{name:>15} seed {seed}: test RMSE {test['rmse']:,.0f} | R² {test['r2']:.4f}")
        runs.append({"seed": seed, "test": test, "history": history})
    return {
        "parameters": count_parameters(model),
        "rmse": summarise([r["test"]["rmse"] for r in runs]),
        "r2": summarise([r["test"]["r2"] for r in runs]),
        "bias": summarise([r["test"]["bias"] for r in runs]),
        "final_train_rmse": summarise([r["history"]["train_rmse"][-1] for r in runs]),
        "runs": runs,
    }


def plot_bars(results, labels, path, title):
    """Mean ± sample std of test RMSE and R², with each seed shown as a dot."""
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, metric, ylabel in [(axes[0], "rmse", "Test RMSE (SGD)"), (axes[1], "r2", "Test R²")]:
        for i, name in enumerate(names):
            stats = results[name][metric]
            ax.bar(i, stats["mean"], width=0.6, color=COLOURS[i], alpha=0.85,
                   yerr=stats["std"], capsize=4, error_kw={"elinewidth": 1.2})
            jitter = np.linspace(-0.12, 0.12, len(stats["values"]))
            ax.scatter(i + jitter, stats["values"], s=14, color="#0b0b0b", zorder=3)
            text = f"{stats['mean'] / 1000:,.1f}k" if metric == "rmse" else f"{stats['mean']:.3f}"
            top = max(stats["mean"] + stats["std"], max(stats["values"]))
            ax.annotate(text, (i, top), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)), [labels[n] for n in names], fontsize=8)
        style_axes(ax, "", ylabel)
        if metric == "rmse":
            thousands(ax)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.08)
    axes[1].set_ylim(0, 1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mean_curves(results, labels, metric, ylabel, path, title):
    """Mean ± std across seeds of the per-epoch test metric (dashed: train)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for colour, (name, result) in zip(COLOURS, results.items()):
        for split, style in [("eval", "-"), ("train", "--")]:
            curves = np.array([r["history"][f"{split}_{metric}"] for r in result["runs"]])
            mean, std = curves.mean(0), curves.std(0, ddof=1)
            epochs = np.arange(1, len(mean) + 1)
            label = f"{labels[name]} – {'test' if split == 'eval' else 'train'}"
            ax.plot(epochs, mean, style, color=colour, linewidth=2, label=label,
                    marker="o" if split == "eval" else None, markersize=4)
            if split == "eval":
                ax.fill_between(epochs, mean - std, mean + std, color=colour, alpha=0.15)
    style_axes(ax, "Epoch", ylabel, title)
    if metric == "rmse":
        thousands(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config, epochs = load_selected()
    print(f"A1 selection: width {config.hidden_width}, embeddings {config.embedding_dims}, "
          f"{epochs} epochs; seeds {SEEDS}")
    data = prepare_data(DATA_PATH, combined_config(config))

    results = {name: run_seeds(name, variant, epochs, data)
               for name, variant in variants(config).items()}
    save_json(results, OUT / "results.json")

    plot_bars(results, LABELS, OUT / "ablation_bars.png", "A2 ablations: test metrics over 5 seeds")
    plot_mean_curves(results, LABELS, "rmse", "RMSE (SGD)", OUT / "ablation_rmse_curves.png",
                     "RMSE per epoch (mean ± std over 5 seeds)")
    plot_mean_curves(results, LABELS, "r2", "R²", OUT / "ablation_r2_curves.png",
                     "R² per epoch (mean ± std over 5 seeds)")

    print("\nvariant           params   test RMSE (mean ± std)   test R² (mean ± std)   bias")
    for name, r in results.items():
        print(f"{name:<16} {r['parameters']:>7,}   {r['rmse']['mean']:>9,.0f} ± {r['rmse']['std']:>6,.0f}"
              f"        {r['r2']['mean']:.4f} ± {r['r2']['std']:.4f}   {r['bias']['mean']:>8,.0f}")


if __name__ == "__main__":
    main()
