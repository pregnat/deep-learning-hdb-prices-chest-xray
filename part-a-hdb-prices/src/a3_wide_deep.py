"""Question A3: wide-and-deep model, output = MLP(x) + λ · Linear(x).

1. λ search over {0, 0.25, 0.5, 1, 2}: train on 2017-2020 with early stopping on 2021
   (seed 42, A1 hyperparameters fixed) and pick the lowest best validation RMSE.
2. Retrain the selected λ on 2017-2021 with the five A2 seeds and the A1 epoch count,
   evaluate once on 2022 and compare with the A2 baseline runs (same seeds).

Requires outputs/a2/results.json. Run from the repository root:
    uv run python src/a3_wide_deep.py
"""

import dataclasses
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from a2_ablation import plot_bars, plot_mean_curves, run_seeds
from common import (
    COLOURS, DATA_PATH, OUTPUT_DIR, SEEDS, build_model, combined_config, load_selected,
    plot_curves, run_final, save_json, set_seed, style_axes, thousands, train_model,
)
from data import prepare_data

OUT = OUTPUT_DIR / "a3"
LAMBDAS = [0, 0.25, 0.5, 1, 2]
SEARCH_SEED = 42


def lambda_search(config):
    data = prepare_data(DATA_PATH, config)
    rows, curves = [], {}
    for lam in LAMBDAS:
        set_seed(SEARCH_SEED)
        model = build_model(data, dataclasses.replace(config, wide_lambda=lam))
        history = train_model(model, data["train"], data["validation"], config)
        best = history["best_epoch"]
        rows.append({"lambda": lam, "best_validation_rmse": history["eval_rmse"][best - 1],
                     "best_epoch": best, "epochs_run": len(history["eval_rmse"])})
        curves[str(lam)] = history["eval_rmse"]
        print(f"λ = {lam:<4}: best validation RMSE {rows[-1]['best_validation_rmse']:,.0f} "
              f"(epoch {best})")
    return rows, curves


@torch.no_grad()
def branch_contributions(model, dataset):
    """Std (SGD) of each branch's contribution to the prediction on a dataset."""
    model.eval()
    categorical, continuous, _ = dataset.tensors
    embedded = [emb(categorical[:, i]) for i, emb in enumerate(model.embeddings)]
    features = torch.cat(embedded + [continuous], dim=1)
    deep = model.mlp(features).squeeze(1) * model.target_std
    wide = model.wide_lambda * model.linear(features).squeeze(1) * model.target_std
    return {"deep_std": deep.std().item(), "wide_std": wide.std().item(),
            "correlation": torch.corrcoef(torch.stack([deep, wide]))[0, 1].item()}


def plot_lambda_search(rows, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    best = min(rows, key=lambda r: r["best_validation_rmse"])
    for i, r in enumerate(rows):
        colour = COLOURS[0] if r is best else "#a9c6ec"
        ax.bar(i, r["best_validation_rmse"], width=0.6, color=colour)
        ax.annotate(f"{r['best_validation_rmse'] / 1000:.1f}k\n(ep {r['best_epoch']})",
                    (i, r["best_validation_rmse"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)), [f"λ = {r['lambda']}" for r in rows])
    style_axes(ax, "", "Best validation RMSE (SGD)", "λ search (validation 2021, seed 42)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    thousands(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_paired(baseline, wide, path):
    """Per-seed test RMSE of baseline vs wide-and-deep (paired by seed)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    b = [r["test"]["rmse"] for r in baseline["runs"]]
    w = [r["test"]["rmse"] for r in wide["runs"]]
    for seed, y0, y1 in zip(SEEDS, b, w):
        ax.plot([0, 1], [y0, y1], color="#a3a29d", linewidth=1, zorder=1)
    ax.scatter([0] * len(b), b, color=COLOURS[0], s=36, zorder=2, label="Baseline")
    ax.scatter([1] * len(w), w, color=COLOURS[1], s=36, zorder=2, label="Wide & deep")
    ax.set_xticks([0, 1], ["Baseline", "Wide & deep"])
    ax.set_xlim(-0.4, 1.4)
    style_axes(ax, "", "Test RMSE (SGD)", "Per-seed test RMSE (lines join same seed)")
    thousands(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config, epochs = load_selected()

    # 1. λ search on the validation year.
    rows, curves = lambda_search(config)
    best = min(rows, key=lambda r: r["best_validation_rmse"])
    plot_lambda_search(rows, OUT / "lambda_search.png")
    plot_curves([(f"λ = {k}", v) for k, v in curves.items()], "Validation RMSE (SGD)",
                OUT / "lambda_validation_curves.png", "Validation RMSE per epoch for each λ")
    print(f"selected λ = {best['lambda']}")

    # 2. Five seeds on train + validation, compared with the A2 baseline (same seeds).
    wide_config = dataclasses.replace(config, wide_lambda=best["lambda"])
    data = prepare_data(DATA_PATH, combined_config(config))
    wide = run_seeds("wide_and_deep", wide_config, epochs, data)
    baseline = json.loads((OUTPUT_DIR / "a2" / "results.json").read_text())["baseline"]

    # Size of each branch's contribution on the test set (seed 42 model).
    model, _, _ = run_final(wide_config, SEEDS[0], epochs, data)
    torch.save(model.state_dict(), OUT / "wide_deep_seed42.pt")
    contributions = branch_contributions(model, data["test"]) if best["lambda"] else None

    diffs = np.array([w["test"]["rmse"] - b["test"]["rmse"]
                      for w, b in zip(wide["runs"], baseline["runs"])])
    comparison = {
        "rmse_difference_per_seed": diffs.tolist(),
        "rmse_difference_mean": float(diffs.mean()),
        "rmse_difference_std": float(diffs.std(ddof=1)),
        "seeds_where_wide_better": int((diffs < 0).sum()),
    }
    save_json({"lambda_search": rows, "selected_lambda": best["lambda"],
               "wide_and_deep": wide, "comparison": comparison,
               "branch_contributions_seed42": contributions}, OUT / "results.json")

    labels = {"baseline": "Baseline (A1)", "wide_and_deep": f"Wide & deep (λ={best['lambda']})"}
    both = {"baseline": baseline, "wide_and_deep": wide}
    plot_bars(both, labels, OUT / "wide_vs_baseline_bars.png",
              "Baseline vs wide-and-deep: test metrics over 5 seeds")
    plot_mean_curves(both, labels, "rmse", "RMSE (SGD)", OUT / "wide_vs_baseline_rmse_curves.png",
                     "RMSE per epoch (mean ± std over 5 seeds)")
    plot_mean_curves(both, labels, "r2", "R²", OUT / "wide_vs_baseline_r2_curves.png",
                     "R² per epoch (mean ± std over 5 seeds)")
    plot_paired(baseline, wide, OUT / "paired_seed_rmse.png")

    for name, r in both.items():
        print(f"{name:<14} params {r['parameters']:>6,} | RMSE {r['rmse']['mean']:,.0f} ± "
              f"{r['rmse']['std']:,.0f} | R² {r['r2']['mean']:.4f} ± {r['r2']['std']:.4f} | "
              f"bias {r['bias']['mean']:,.0f}")
    print(f"paired RMSE difference (wide - baseline): {diffs.mean():,.0f} ± {diffs.std(ddof=1):,.0f}; "
          f"wide better on {comparison['seeds_where_wide_better']}/5 seeds")
    print(f"branch contributions (seed 42, test): {contributions}")


if __name__ == "__main__":
    main()
