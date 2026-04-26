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
python -m pip install -r requirements.txt
```

Validate configured inputs:

```powershell
python -m nuris_pipeline.cli validate-inputs --config configs/v1_inference.yaml
```

Run inference:

```powershell
python -m nuris_pipeline.cli run-inference --config configs/v1_inference.yaml
```

## Outputs

- `outputs/detections.geojson`
- `outputs/zone_stats.csv`
- `outputs/run_manifest.json`

## Known Limitations

- The shipped backend is heuristic, not a tuned satellite model.
- Road extraction is best treated as major-road baseline quality until fine-tuned models are added.
- Missing raster georeferencing is not repaired automatically in v1.
