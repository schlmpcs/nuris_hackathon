# NURIS Satellite Object Detection Prototype

This repository contains a first-version geospatial pipeline for extracting `buildings`, `roads`, and `water` from provided satellite imagery inside a supplied AOI.

## v1 Scope

- GeoTIFF ingestion
- AOI-aware processing
- tiled inference
- heuristic baseline segmentation backend
- vectorization into GIS features
- GeoJSON export
- zone-level summary metrics
- QA control-sample utilities

The current model backend is a heuristic baseline. The code is structured so pretrained segmentation checkpoints can be added later through `src/nuris_pipeline/models/model_registry.py`.

## Quick Start

Install dependencies:

```powershell
py -3.11 -m pip install -r requirements.txt
```

Download `LandCover.ai` from Hugging Face:

```powershell
py -3.11 -m pip install huggingface_hub
py -3.11 -m nuris_pipeline.cli download-landcover-ai --output-dir data/landcover_ai
```

Validate configured inputs:

```powershell
py -3.11 -m nuris_pipeline.cli validate-inputs --config configs/v1_inference.yaml
```

Run inference:

```powershell
py -3.11 -m nuris_pipeline.cli run-inference --config configs/v1_inference.yaml
```

Prepare a `LandCover.ai` dataset manifest:

```powershell
py -3.11 -m nuris_pipeline.cli prepare-landcover-ai --dataset-root data/landcover_ai --output data/landcover_ai_manifest.json
```

Expected `LandCover.ai` layout:

```text
data/landcover_ai/
  images/
  masks/
  train.txt
  val.txt
  test.txt
```

`LandCover.ai` uses `EPSG:2180` and class ids `building=1`, `woodland=2`, `water=3`, `road=4`. For NURIS v1, the dataset integration remaps masks to `background=0`, `building=1`, `water=2`, `road=3`, and drops `woodland`.

Prepare split-aware training patches that match the upstream `split.py` workflow:

```powershell
py -3.11 -m nuris_pipeline.cli prepare-landcover-ai-patches --dataset-root data/landcover_ai --output-dir data/landcover_ai_patches --manifest-output data/landcover_ai_patches/manifest.json
```

This command:

- cuts raw orthophotos into non-overlapping `512x512` patches
- writes images as `<scene>_<k>.jpg`
- writes remapped masks as `<scene>_<k>_m.png`
- writes a patch manifest with `train`, `validation`, and `test` splits based on the published LandCover.ai split files

Train the first segmentation model from the prepared manifest:

```powershell
py -3.11 -m nuris_pipeline.cli train-segmentation --config configs/landcover_ai_training.yaml
```

This training command:

- reads `data/landcover_ai_patches/manifest.json`
- trains a 4-class `U-Net` on the `train` split
- validates on `validation`
- evaluates on `test`
- saves `best.pt`, `final.pt`, `metrics.csv`, `metrics.json`, and a config copy under `outputs/checkpoints/`

Training-ready configuration is staged in [landcover_ai_training.yaml](/C:/Users/izimo/Desktop/Amir/nuris_hackathon/configs/landcover_ai_training.yaml).

# nuris_hackathon
```

## Outputs

- `outputs/detections.geojson`
- `outputs/zone_stats.csv`
- `outputs/run_manifest.json`

## Known Limitations

- The shipped backend is heuristic, not a tuned satellite model.
- Road extraction is best treated as major-road baseline quality until fine-tuned models are added.
- Missing raster georeferencing is not repaired automatically in v1.
