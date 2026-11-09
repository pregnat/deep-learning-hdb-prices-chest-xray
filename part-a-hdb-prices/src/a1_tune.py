"""Question A1: baseline diagram/parameters, Ray Tune grid search, final retrain.

Run from the repository root:  uv run python src/a1_tune.py
Outputs go to outputs/a1/.
"""

import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from matplotlib.patches import FancyBboxPatch
from ray import tune

from common import (
    COLOURS, DATA_PATH, OUTPUT_DIR, build_model, count_parameters, plot_curves,
    run_final, save_json, set_seed, train_model,
)
from config import Config
from data import prepare_data

OUT = OUTPUT_DIR / "a1"
SEED = 42
HIDDEN_WIDTHS = [16, 32, 64, 128, 256]
EMBEDDING_DIMS = [2, 4, 8, 12, 16]


def parameter_breakdown(model):
    return {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}


def draw_block_diagram(config, data, path):
    """Block diagram of the baseline (default Config: embedding 8, hidden 64)."""
    names = config.categorical_features
    cards = data["cardinalities"]
    emb = config.embedding_dims
    n_cont = len(config.continuous_features)
    concat = sum(emb) + n_cont

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    def box(x, y, w, h, text, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=colour, edgecolor="#52514e", linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#52514e", lw=1))

    # Inputs and embeddings
    xs = [0.1, 1.75, 3.4, 5.05]
    for x, name, card, dim in zip(xs, names, cards, emb):
        box(x, 9.2, 1.55, 0.8, f"{name}\n(int index)", "#e9f1fb")
        box(x, 7.7, 1.55, 0.9, f"Embedding\n{card} × {dim}", "#fde9e0")
        arrow(x + 0.775, 9.2, x + 0.775, 8.62)
        arrow(x + 0.775, 7.7, 4.6, 6.62)
    box(7.0, 8.45, 2.8, 1.55, "6 continuous features\n(standardised)\ndist_to_nearest_stn, dist_to_dhoby,\n"
        "degree_c., eigenvector_c.,\nremaining_lease, floor_area", "#e9f1fb")
    arrow(8.4, 8.45, 4.9, 6.62)

    box(2.9, 5.8, 3.6, 0.8, f"Concatenate → {concat}-dim vector\n({sum(emb)} embedding + {n_cont} continuous)",
        "#f1f0ec")
    box(2.9, 4.4, 3.6, 0.8, f"Linear {concat} → {config.hidden_width}", "#e2f5ee")
    box(2.9, 3.2, 3.6, 0.7, f"LayerNorm({config.hidden_width})", "#e2f5ee")
    box(2.9, 2.0, 3.6, 0.7, "ReLU", "#e2f5ee")
    box(2.9, 0.8, 3.6, 0.7, f"Linear {config.hidden_width} → 1", "#e2f5ee")
    for y0, y1 in [(5.8, 5.22), (4.4, 3.92), (3.2, 2.72), (2.0, 1.52)]:
        arrow(4.7, y0, 4.7, y1)
    box(7.0, 0.8, 2.8, 0.7, "× target std + target mean\n→ resale price (SGD)", "#fff4d6")
    arrow(6.5, 1.15, 7.0, 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def trainable(params, data):
    """One Ray Tune trial: train with early stopping on 2021 validation data."""
    torch.set_num_threads(1)
    config = dataclasses.replace(
        Config(),
        hidden_width=params["hidden_width"],
        embedding_dims=(params["embedding_dim"],) * 4,
    )
    set_seed(SEED)
    model = build_model(data, config)
    best = {"rmse": float("inf"), "epoch": 0}

    def report(epoch, metrics):
        if metrics["eval_rmse"] < best["rmse"]:
            best.update(rmse=metrics["eval_rmse"], epoch=epoch)
        tune.report({
            "epoch": epoch,
            "train_rmse": metrics["train_rmse"],
            "validation_rmse": metrics["eval_rmse"],
            "best_validation_rmse": best["rmse"],
            "best_epoch": best["epoch"],
            "parameters": count_parameters(model),
        })

    train_model(model, data["train"], data["validation"], config, on_epoch=report)


def grid_search(data):
    ray.init(num_cpus=os.cpu_count(), include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": str(OUTPUT_DIR.parent / "src")}})
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainable, data=data), {"cpu": 1}),
        param_space={
            "hidden_width": tune.grid_search(HIDDEN_WIDTHS),
            "embedding_dim": tune.grid_search(EMBEDDING_DIMS),
        },
        tune_config=tune.TuneConfig(metric="best_validation_rmse", mode="min"),
        run_config=tune.RunConfig(name="a1_grid", storage_path=str(OUT / "ray_results"),
                                  verbose=1),
    )
    results = tuner.fit()
    ray.shutdown()

    rows = []
    curves = {}
    for result in results:
        frame = result.metrics_dataframe
        h, e = result.config["hidden_width"], result.config["embedding_dim"]
        last = frame.iloc[-1]
        rows.append({
            "hidden_width": h, "embedding_dim": e,
            "best_validation_rmse": float(last["best_validation_rmse"]),
            "best_epoch": int(last["best_epoch"]),
            "epochs_run": int(last["epoch"]),
            "train_rmse_at_best": float(frame["train_rmse"].iloc[int(last["best_epoch"]) - 1]),
            "parameters": int(last["parameters"]),
        })
        curves[f"{h}_{e}"] = {"train_rmse": frame["train_rmse"].tolist(),
                              "validation_rmse": frame["validation_rmse"].tolist()}
    rows.sort(key=lambda r: r["best_validation_rmse"])
    return rows, curves


def plot_grid(rows, path):
    grid = np.zeros((len(HIDDEN_WIDTHS), len(EMBEDDING_DIMS)))
    for r in rows:
        grid[HIDDEN_WIDTHS.index(r["hidden_width"]), EMBEDDING_DIMS.index(r["embedding_dim"])] = \
            r["best_validation_rmse"]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    image = ax.imshow(grid / 1000, cmap="Blues_r")
    ax.set_xticks(range(len(EMBEDDING_DIMS)), EMBEDDING_DIMS)
    ax.set_yticks(range(len(HIDDEN_WIDTHS)), HIDDEN_WIDTHS)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Hidden-layer width")
    best = rows[0]
    for i in range(len(HIDDEN_WIDTHS)):
        for j in range(len(EMBEDDING_DIMS)):
            is_best = (HIDDEN_WIDTHS[i], EMBEDDING_DIMS[j]) == (best["hidden_width"], best["embedding_dim"])
            ax.text(j, i, f"{grid[i, j] / 1000:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_best else "normal",
                    color="white" if grid[i, j] < np.median(grid) else "#0b0b0b")
    fig.colorbar(image, ax=ax, label="Best validation RMSE (SGD thousands)")
    ax.set_title("Grid search: best validation RMSE", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_config = Config()
    data = prepare_data(DATA_PATH, base_config)

    # 1. Baseline block diagram and trainable parameters (default config).
    set_seed(SEED)
    baseline = build_model(data, base_config)
    draw_block_diagram(base_config, data, OUT / "baseline_block_diagram.png")
    summary = {
        "baseline_parameters": count_parameters(baseline),
        "baseline_parameter_breakdown": parameter_breakdown(baseline),
        "cardinalities": dict(zip(base_config.categorical_features, data["cardinalities"])),
        "split_sizes": {k: len(data[k]) for k in ("train", "validation", "test")},
    }
    print(f"baseline trainable parameters: {summary['baseline_parameters']:,}")

    # 2. Grid search with Ray Tune.
    rows, curves = grid_search(data)
    best = rows[0]
    save_json(rows, OUT / "grid_results.json")
    save_json(curves, OUT / "grid_curves.json")
    plot_grid(rows, OUT / "grid_heatmap.png")
    print("\nTop 5 configurations:")
    for r in rows[:5]:
        print(f"  width {r['hidden_width']:>3}, emb {r['embedding_dim']:>2}: "
              f"best val RMSE {r['best_validation_rmse']:,.0f} (epoch {r['best_epoch']})")

    # 3. Learning curves of the selected configuration.
    selected = curves[f"{best['hidden_width']}_{best['embedding_dim']}"]
    plot_curves([("Train", selected["train_rmse"]), ("Validation (2021)", selected["validation_rmse"])],
                "RMSE (SGD)", OUT / "selected_train_val_rmse.png",
                f"Selected config (width {best['hidden_width']}, embedding {best['embedding_dim']})")

    # 4. Retrain on 2017-2021 for the selected number of epochs, test on 2022.
    config = dataclasses.replace(base_config, hidden_width=best["hidden_width"],
                                 embedding_dims=(best["embedding_dim"],) * 4)
    model, history, test = run_final(config, SEED, best["best_epoch"])
    torch.save(model.state_dict(), OUT / "final_model.pt")
    plot_curves([("Train (2017-2021)", history["train_rmse"]), ("Test (2022)", history["eval_rmse"])],
                "RMSE (SGD)", OUT / "final_train_test_rmse.png", "Final model retrained on train + validation")

    summary.update({
        "selected": best,
        "selected_parameters": count_parameters(model),
        "selected_config": {"hidden_width": config.hidden_width,
                            "embedding_dims": list(config.embedding_dims),
                            "epochs": best["best_epoch"]},
        "final_history": history,
        "final_test": test,
    })
    save_json(summary, OUT / "summary.json")
    print(f"\nSelected: hidden width {best['hidden_width']}, embedding dim {best['embedding_dim']}, "
          f"epochs {best['best_epoch']}")
    print(f"Final test RMSE {test['rmse']:,.0f} | R² {test['r2']:.4f}")


if __name__ == "__main__":
    main()
