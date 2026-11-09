"""Question A4: Captum FeatureAblation on the 2022 test set.

Explains the final A1 baseline (outputs/a1/final_model.pt) and the A3 wide-and-deep
model (outputs/a3/wide_deep_seed42.pt); both were trained on 2017-2021 with seed 42.

Each of the ten input features is ablated as one group:
  - categorical features -> index 0, the "unseen category" ID reserved in data.py
  - continuous features  -> 0, i.e. the training mean after standardisation
The attribution of a feature for a sample is f(x) - f(x with that feature ablated),
in Singapore dollars (the forward function returns predict_price). Global importance
is the mean absolute attribution over all test samples.

Run from the repository root:  uv run python src/a4_captum.py
"""

import dataclasses
import json

import matplotlib.pyplot as plt
import torch
from captum.attr import FeatureAblation

from common import (
    COLOURS, DATA_PATH, OUTPUT_DIR, build_model, combined_config, load_selected,
    save_json, set_seed, style_axes,
)
from data import prepare_data

OUT = OUTPUT_DIR / "a4"
BATCH_SIZE = 2048


def feature_importance(model, dataset, feature_names):
    model.eval()
    explainer = FeatureAblation(model.predict_price)
    categorical, continuous, _ = dataset.tensors
    n_cat = categorical.shape[1]
    mask = (torch.arange(n_cat).unsqueeze(0),
            torch.arange(n_cat, n_cat + continuous.shape[1]).unsqueeze(0))

    total = torch.zeros(len(feature_names))
    for start in range(0, len(categorical), BATCH_SIZE):
        batch = (categorical[start:start + BATCH_SIZE], continuous[start:start + BATCH_SIZE])
        with torch.no_grad():
            cat_attr, cont_attr = explainer.attribute(
                batch, baselines=(0, 0.0), feature_mask=mask, perturbations_per_eval=4
            )
        # Every column in a group carries that group's attribution; take one per feature.
        attribution = torch.cat([cat_attr, cont_attr], dim=1).float()
        total += attribution.abs().sum(0)
    return dict(zip(feature_names, (total / len(categorical)).tolist()))


def plot_importance(importance, title, colour, path):
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names, values = zip(*items)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = range(len(names))
    ax.barh(y, values, color=colour, height=0.65)
    for i, v in enumerate(values):
        ax.annotate(f"S${v:,.0f}", (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    style_axes(ax, "Mean |attribution| on 2022 test set (S$)", "", title)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_selected()
    a3 = json.loads((OUTPUT_DIR / "a3" / "results.json").read_text())
    data = prepare_data(DATA_PATH, combined_config(config))
    features = list(config.categorical_features) + list(config.continuous_features)

    models = {
        "baseline": (config, OUTPUT_DIR / "a1" / "final_model.pt", "Baseline (A1)"),
        "wide_and_deep": (dataclasses.replace(config, wide_lambda=a3["selected_lambda"]),
                          OUTPUT_DIR / "a3" / "wide_deep_seed42.pt",
                          f"Wide & deep (λ={a3['selected_lambda']})"),
    }
    results = {}
    for colour, (name, (model_config, path, label)) in zip(COLOURS, models.items()):
        set_seed(42)
        model = build_model(data, model_config)
        model.load_state_dict(torch.load(path))
        importance = feature_importance(model, data["test"], features)
        ranked = sorted(importance, key=importance.get, reverse=True)
        results[name] = {"importance_sgd": importance, "ranking": ranked}
        plot_importance(importance, f"FeatureAblation importance – {label}", colour,
                        OUT / f"importance_{name}.png")
        print(f"\n{label}: top 3 = {ranked[:3]}")
        for feature in ranked:
            print(f"  {feature:<24} S${importance[feature]:>10,.0f}")
    save_json(results, OUT / "results.json")


if __name__ == "__main__":
    main()
