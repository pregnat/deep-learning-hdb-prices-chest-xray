"""Question B3: ResNet-like CNN with the B1 learning rate and B2 batch size, plus the
comparison of all three architectures (accuracy, sensitivity, specificity, error,
parameters, FLOPs).

Requires outputs/b1/results.json and outputs/b2/results.json.
Run from the PartB folder:  uv run python src/b3_resnet.py
"""

import json

import torch

from common import (
    OUTPUT_DIR, build_and_train, count_flops, count_parameters, evaluate, load_data,
    plot_validation_curves, save_json,
)
from models import MODELS

OUT = OUTPUT_DIR / "b3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    b1 = json.loads((OUTPUT_DIR / "b1" / "results.json").read_text())
    b2 = json.loads((OUTPUT_DIR / "b2" / "results.json").read_text())
    lr, batch_size = b1["selected_learning_rate"], b2["selected_batch_size"]

    model, result = build_and_train("resnet", data, lr, batch_size)
    test = evaluate(model, data["test"])
    torch.save(model.state_dict(), OUT / "resnet.pt")
    plot_validation_curves({"ResNet-like": result}, "val_loss", "Validation loss (BCE)",
                           "B3 ResNet-like: validation loss", OUT / "val_loss.png")

    # Comparison table: test metrics of the three selected models.
    rows = []
    for name, label, metrics, setting in [
        ("alexnet", "B1 AlexNet-like", b1["test"], f"lr {lr:g}, batch 128"),
        ("vgg", "B2 VGG-like", b2["test"], f"lr {lr:g}, batch {batch_size}"),
        ("resnet", "B3 ResNet-like", test, f"lr {lr:g}, batch {batch_size}"),
    ]:
        architecture = MODELS[name]()
        rows.append({
            "model": label, "training": setting,
            "accuracy": metrics["accuracy"], "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"], "error": metrics["error"],
            "parameters": count_parameters(architecture), "flops": count_flops(architecture),
        })
    save_json({"learning_rate": lr, "batch_size": batch_size, "best_epoch": result["best_epoch"],
               "validation": result["validation"], "test": test, "comparison": rows},
              OUT / "results.json")

    print(f"\nResNet-like test: accuracy {test['accuracy']:.4f} | sensitivity "
          f"{test['sensitivity']:.4f} | specificity {test['specificity']:.4f}")
    print(f"\n{'model':<16} | accuracy | sensitivity | specificity | error  | params | MFLOPs")
    for r in rows:
        print(f"{r['model']:<16} | {r['accuracy']:.4f}   | {r['sensitivity']:.4f}      | "
              f"{r['specificity']:.4f}      | {r['error']:.4f} | {r['parameters']:>6,} | "
              f"{r['flops'] / 1e6:.2f}")


if __name__ == "__main__":
    main()
