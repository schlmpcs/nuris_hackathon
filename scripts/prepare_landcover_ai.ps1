$ErrorActionPreference = "Stop"

python -m nuris_pipeline.cli prepare-landcover-ai --dataset-root data/landcover_ai --output data/landcover_ai_manifest.json
