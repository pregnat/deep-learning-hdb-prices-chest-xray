# SC4001 Part B: chest X-ray classification

### Environment and dependencies
- Python 3.12 (`.python-version`), managed with [uv](https://docs.astral.sh/uv/).
- Dependencies are pinned in `uv.lock` (torch 2.13, captum 0.9, numpy 2, matplotlib 3.11).
  Install with `uv sync`.
- Tested on macOS (Apple M3); all training runs on CPU for reproducibility.
- `chestxray_binary_64.npz` must be in this folder.

### Seed
- Seed `42` for model initialisation, data shuffling and the random choice of Grad-CAM images.
- `torch.use_deterministic_algorithms(True)` is enabled.

### Commands (run in order, from this folder)
```bash
uv sync
uv run python src/b1_alexnet.py   # B1: AlexNet-like, learning rates {1e-5 … 1e-1}, batch 128
uv run python src/b2_vgg.py       # B2: VGG-like, batch sizes {16 … 256}, best B1 learning rate
uv run python src/b3_resnet.py    # B3: ResNet-like + comparison table (params, FLOPs)
uv run python src/b4_gradcam.py   # B4: Grad-CAM overlays on 5 random pneumonia test X-rays
```
Each later script reads the earlier results from `outputs/`. Plots, model weights and
JSON results are written to `outputs/b1` … `outputs/b4`.

### Protocol
- Images scaled to [0, 1], then standardised with the training-set mean and std.
- Label 1 = pneumonia (positive class): sensitivity = pneumonia recall,
  specificity = normal recall.
- Binary cross-entropy on one output logit; Adam; max 20 epochs; early stopping on
  validation loss (patience 4, min improvement 1e-4), best weights restored.
- Model selection uses the lowest validation classification error (ties: lower validation loss).
- Convolutions are bias-free with BatchNorm + ReLU; fully connected layers use bias;
  dropout p = 0.5 before every fully connected layer that follows a hidden representation.
- FLOPs counted with `torch.utils.flop_counter.FlopCounterMode` for one 64×64 image.

### Files
```text
src/models.py       AlexNet-like, VGG-like and ResNet-like architectures (Figure B)
src/common.py       data loading, training, metrics, parameter/FLOP counts, plots
src/b1_alexnet.py   question B1
src/b2_vgg.py       question B2
src/b3_resnet.py    question B3
src/b4_gradcam.py   question B4
```
