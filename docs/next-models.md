# Next Training Models

This note records the planned model sequence after the current `U-Net` baseline.

## Current Baseline

- `U-Net`
- Purpose: establish a simple, trainable 4-class segmentation baseline for `background`, `building`, `water`, and `road`
- Status: current training run

## Next Models To Use

### 1. `SegFormer`

`SegFormer` should be the next primary model.

Why:

- strong semantic segmentation performance for remote-sensing style imagery
- good balance between accuracy and implementation complexity
- transformer backbone is a meaningful upgrade over the current plain `U-Net`
- flexible for later checkpoint-based inference integration

Recommended role:

- first serious quality upgrade after the baseline
- main candidate for replacing the heuristic backend in the NURIS inference pipeline

### 2. `DeepLabV3+`

`DeepLabV3+` should be the next comparison model after `SegFormer`.

Why:

- established segmentation architecture with strong baseline behavior
- often performs well on boundaries and multi-scale context
- lower integration risk than more experimental alternatives

Recommended role:

- benchmark against `SegFormer`
- fallback production model if it is easier to train or deploy reliably

## Proposed Order

1. finish and evaluate `U-Net`
2. train `SegFormer`
3. train `DeepLabV3+`
4. compare validation and test metrics, especially mean IoU for `building`, `water`, and `road`
5. promote the best checkpoint into the inference model registry

## Decision Rule

Prefer the model that gives the best overall GIS-useful segmentation quality under these constraints:

- stable 4-class outputs
- acceptable training time
- fits target GPU memory
- integrates cleanly with the existing checkpoint and inference path
