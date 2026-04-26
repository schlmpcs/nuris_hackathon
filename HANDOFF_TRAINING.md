# Training Handoff

## Objective

Continue this project on another device starting at the point just before model training.

The current repo state includes:

- the NURIS geospatial prototype scaffold
- LandCover.ai dataset integration
- raw dataset download support
- exact upstream-compatible `512x512` patch preparation
- remapped masks for NURIS target classes
- split-aware training manifest generation

## Current Status

Implemented and verified locally:

- `21` tests passing
- `LandCover.ai` raw dataset downloaded
- `LandCover.ai` training patches prepared
- official split alignment verified

Prepared dataset artifacts:

- `data/landcover_ai/`
- `data/landcover_ai_patches/images/`
- `data/landcover_ai_patches/masks/`
- `data/landcover_ai_patches/manifest.json`

Patch manifest counts:

- `train: 7470`
- `validation: 1602`
- `test: 1602`

## Important Files

- `src/nuris_pipeline/data/landcover_ai.py`
- `src/nuris_pipeline/cli.py`
- `configs/landcover_ai_training.yaml`
- `scripts/download_landcover_ai.ps1`
- `scripts/prepare_landcover_ai.ps1`
- `scripts/prepare_landcover_ai_patches.ps1`

## Label Mapping

LandCover.ai original labels:

- `0 = background`
- `1 = building`
- `2 = woodland`
- `3 = water`
- `4 = road`

NURIS training remap used here:

- `0 = background`
- `1 = building`
- `2 = water`
- `3 = road`

`woodland` is dropped to background for this first training pass.

## What The Next AI Should Do

1. Set up the Python environment and install dependencies from `requirements.txt`.
2. Confirm access to `data/landcover_ai_patches/manifest.json`.
3. Implement the actual training pipeline.

Recommended first training scope:

- semantic segmentation only
- `DeepLabV3+`, `SegFormer`, or `U-Net`
- train on the remapped 4-class labels
- use the prepared train/validation/test split directly from the manifest

## Recommended Training Tasks

1. Add a patch dataset loader that reads:
   - `image_path`
   - `mask_path`
   - `split`

2. Add augmentations for training only:
   - horizontal/vertical flips
   - 90-degree rotations
   - mild color jitter

3. Normalize RGB input consistently.

4. Implement:
   - training dataloader
   - validation dataloader
   - loss function for multiclass segmentation
   - checkpoint saving
   - metric logging

5. Use at minimum:
   - mean IoU
   - per-class IoU
   - pixel accuracy

6. Save:
   - best checkpoint
   - final checkpoint
   - training config copy
   - metrics CSV or JSON

## Constraints To Preserve

- Keep class ids stable:
  - `0 background`
  - `1 building`
  - `2 water`
  - `3 road`
- Do not change patch naming or split interpretation.
- Do not rewrite the prepared manifest unless there is a documented bug.
- Keep inference/export compatibility with the existing NURIS GIS pipeline.

## Useful Commands

Download raw dataset:

```powershell
python -m nuris_pipeline.cli download-landcover-ai --output-dir data/landcover_ai
```

Build raw-scene manifest:

```powershell
python -m nuris_pipeline.cli prepare-landcover-ai --dataset-root data/landcover_ai --output data/landcover_ai_manifest.json
```

Build training patches:

```powershell
python -m nuris_pipeline.cli prepare-landcover-ai-patches --dataset-root data/landcover_ai --output-dir data/landcover_ai_patches --manifest-output data/landcover_ai_patches/manifest.json
```

Run current tests:

```powershell
pytest tests -q
```

## Recommended First Training Deliverable

Produce a first trainable segmentation branch that can:

- read `data/landcover_ai_patches/manifest.json`
- train on `train`
- validate on `validation`
- evaluate on `test`
- save a checkpoint that can later replace the heuristic baseline in the current inference pipeline

## Known Notes

- `data/` is gitignored, so move dataset artifacts separately if needed.
- The current repo stops before training by design.
- The target deployment pipeline still uses a heuristic baseline until a trained model is integrated.
