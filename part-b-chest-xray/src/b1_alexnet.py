"""Question B1: AlexNet-like CNN, learning-rate sweep at batch size 128.

Run from the PartB folder:  uv run python src/b1_alexnet.py
Outputs go to outputs/b1/.
"""

import torch

from common import (
    OUTPUT_DIR, build_and_train, evaluate, load_data, plot_validation_curves,
    save_json, select, sweep_table,
)

OUT = OUTPUT_DIR / "b1"
BATCH_SIZE = 128
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def main():
    data = load_data()
    rows, runs, models = [], {}, {}
    for lr in LEARNING_RATES:
        model, result = build_and_train("alexnet", data, lr, BATCH_SIZE)
        rows.append({"learning_rate": lr, **result})
        runs[f"lr = {lr:g}"] = result
        models[lr] = model

    plot_validation_curves(runs, "val_loss", "Validation loss (BCE)",
                           "B1 AlexNet-like: validation loss (○ = best epoch)", OUT / "val_loss.png")
    plot_validation_curves(runs, "val_accuracy", "Validation accuracy",
                           "B1 AlexNet-like: validation accuracy (○ = best epoch)", OUT / "val_accuracy.png")
    sweep_table(rows, "learning_rate")

    best = select(rows)
    lr = best["learning_rate"]
    test = evaluate(models[lr], data["test"])
    torch.save(models[lr].state_dict(), OUT / "alexnet_best.pt")
    save_json({"batch_size": BATCH_SIZE, "sweep": rows, "selected_learning_rate": lr,
               "test": test}, OUT / "results.json")
    print(f"\nselected learning rate: {lr:g}")
    print(f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} | "
          f"specificity {test['specificity']:.4f}")


if __name__ == "__main__":
    main()
