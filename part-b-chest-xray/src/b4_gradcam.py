"""Question B4: Grad-CAM (Captum LayerGradCam) on five random pneumonia test X-rays.

Uses the trained models from B1-B3 and the last convolutional feature map of each
(`cam_layer` in models.py). Heatmaps are upsampled to 64x64 and overlaid.

Requires outputs/b1, b2, b3 model files. Run from the PartB folder:
    uv run python src/b4_gradcam.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam

from common import OUTPUT_DIR, SEED, load_data, save_json
from models import MODELS

OUT = OUTPUT_DIR / "b4"
N_IMAGES = 5
CHECKPOINTS = {
    "alexnet": ("B1 AlexNet-like", OUTPUT_DIR / "b1" / "alexnet_best.pt"),
    "vgg": ("B2 VGG-like", OUTPUT_DIR / "b2" / "vgg_best.pt"),
    "resnet": ("B3 ResNet-like", OUTPUT_DIR / "b3" / "resnet.pt"),
}


def grad_cam(model, images):
    """ReLU'd Grad-CAM for the pneumonia logit, upsampled and scaled to [0, 1] per image."""
    cam = LayerGradCam(model, model.cam_layer)
    attribution = cam.attribute(images, target=0, relu_attributions=True)
    attribution = LayerAttribution.interpolate(attribution, images.shape[-2:], "bilinear")
    attribution = attribution.squeeze(1).detach()
    peak = attribution.flatten(1).max(1).values.clamp_min(1e-12)
    return (attribution / peak[:, None, None]).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    images, labels = data["test"].tensors
    pneumonia = torch.nonzero(labels == 1).squeeze(1)
    generator = torch.Generator().manual_seed(SEED)
    chosen = pneumonia[torch.randperm(len(pneumonia), generator=generator)[:N_IMAGES]]
    batch = images[chosen]
    display = (batch.squeeze(1) * data["std"] + data["mean"]).clamp(0, 1).numpy()

    heatmaps, probabilities = {}, {}
    for name, (label, path) in CHECKPOINTS.items():
        model = MODELS[name]()
        model.load_state_dict(torch.load(path))
        model.eval()
        with torch.no_grad():
            probabilities[name] = torch.sigmoid(model(batch).squeeze(1)).tolist()
        heatmaps[name] = grad_cam(model, batch)

    columns = len(CHECKPOINTS) + 1
    fig, axes = plt.subplots(N_IMAGES, columns, figsize=(2.3 * columns, 2.3 * N_IMAGES))
    for row in range(N_IMAGES):
        axes[row, 0].imshow(display[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"test #{chosen[row].item()}", fontsize=8)
        for col, (name, (label, _)) in enumerate(CHECKPOINTS.items(), start=1):
            axes[row, col].imshow(display[row], cmap="gray", vmin=0, vmax=1)
            image = axes[row, col].imshow(heatmaps[name][row], cmap="jet", alpha=0.45,
                                          vmin=0, vmax=1)
            axes[row, col].set_xlabel(f"P(pneumonia) = {probabilities[name][row]:.2f}", fontsize=7)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    for col, title in enumerate(["X-ray (pneumonia)"] + [v[0] for v in CHECKPOINTS.values()]):
        axes[0, col].set_title(title, fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="Grad-CAM (normalised)")
    fig.savefig(OUT / "gradcam_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_json({"test_indices": chosen.tolist(), "probabilities": probabilities},
              OUT / "results.json")
    print(f"selected test indices: {chosen.tolist()}")
    for name, probs in probabilities.items():
        print(f"{name:<8} P(pneumonia): {np.round(probs, 3).tolist()}")


if __name__ == "__main__":
    main()
