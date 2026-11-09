"""Question B2: VGG-like CNN, batch-size sweep at the learning rate selected in B1.

Requires outputs/b1/results.json. Run from the PartB folder:  uv run python src/b2_vgg.py
Outputs go to outputs/b2/.
"""

import json

import torch

from common import (
    OUTPUT_DIR, build_and_train, evaluate, load_data, plot_validation_curves,
    save_json, select, sweep_table,
)

OUT = OUTPUT_DIR / "b2"
BATCH_SIZES = [16, 32, 64, 128, 256]


def main():
    data = load_data()
    lr = json.loads((OUTPUT_DIR / "b1" / "results.json").read_text())["selected_learning_rate"]
    rows, runs, models = [], {}, {}
    for batch_size in BATCH_SIZES:
        model, result = build_and_train("vgg", data, lr, batch_size)
        rows.append({"batch_size": batch_size, **result})
        runs[f"batch {batch_size}"] = result
        models[batch_size] = model

    plot_validation_curves(runs, "val_loss", "Validation loss (BCE)",
                           f"B2 VGG-like (lr {lr:g}): validation loss (○ = best epoch)",
                           OUT / "val_loss.png")
    plot_validation_curves(runs, "val_accuracy", "Validation accuracy",
                           f"B2 VGG-like (lr {lr:g}): validation accuracy (○ = best epoch)",
                           OUT / "val_accuracy.png")
    sweep_table(rows, "batch_size")

    best = select(rows)
    batch_size = best["batch_size"]
    test = evaluate(models[batch_size], data["test"])
    torch.save(models[batch_size].state_dict(), OUT / "vgg_best.pt")
    save_json({"learning_rate": lr, "sweep": rows, "selected_batch_size": batch_size,
               "test": test}, OUT / "results.json")
    print(f"\nselected batch size: {batch_size}")
    print(f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} | "
          f"specificity {test['specificity']:.4f}")


if __name__ == "__main__":
    main()
