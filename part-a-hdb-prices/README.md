# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python src/baseline.py
```

Run the command from the repository root. The input CSV and output directory are constants at the top of `src/baseline.py`, while the data split, features, model size, and training hyperparameters are defined in `src/config.py`.

## Code map

```text
src/config.py       run configuration
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings and MLP
src/baseline.py     single-run training, RMSE evaluation, and model export
```

## Part A assignment code (SC4001)

### Environment and dependencies
- Python 3.12 (`.python-version`), managed with [uv](https://docs.astral.sh/uv/).
- Dependencies are pinned in `uv.lock` (torch 2.13, ray[tune] 2.58, captum 0.9,
  pandas 3, numpy 2, matplotlib 3.11). Install with `uv sync`.
- Tested on macOS (Apple M3, 8 CPU cores); all training runs on CPU.

### Seeds
- A1 grid search, A1 final model, A3 λ search: seed `42`.
- A2 and A3 multi-seed runs: seeds `42, 43, 44, 45, 46` (`SEEDS` in `src/common.py`).
- `torch.use_deterministic_algorithms(True)` is enabled.

### Commands (run in order, from this folder)
```bash
uv sync
uv run python src/a1_tune.py       # A1: diagram, parameter count, Ray grid search, final retrain (~2 min)
uv run python src/a2_ablation.py   # A2: baseline vs no-embeddings vs sigmoid, 5 seeds (~2 min)
uv run python src/a3_wide_deep.py  # A3: λ search + wide-and-deep over 5 seeds (~1.5 min)
uv run python src/a4_captum.py     # A4: Captum FeatureAblation importances (<1 min)
```
Each later script reads the earlier results from `outputs/`; plots and JSON results are
written to `outputs/a1` … `outputs/a4`.

### Protocol
- Split: 2017-2020 train, 2021 validation, 2022 test. Adam, max 20 epochs, early stopping
  on validation RMSE (patience 4, min improvement 1e-4).
- Retraining on train + validation (2017-2021) uses the best epoch count found with early
  stopping in A1, since no held-out validation set remains. The test set is only monitored
  for the learning-curve plots and is never used for any selection.
- Encoders and scalers are fitted on whichever years are used for training.

### Added files
```text
src/common.py        seeding, training with history, RMSE/R²/bias, plotting
src/a1_tune.py       question A1
src/a2_ablation.py   question A2
src/a3_wide_deep.py  question A3
src/a4_captum.py     question A4
src/model.py         extended with ablation (no embeddings, sigmoid) and wide-and-deep options
src/config.py        extended with the matching options (defaults = original baseline)
```
