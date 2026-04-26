$ErrorActionPreference = "Stop"

python -m nuris_pipeline.cli prepare-landcover-ai-patches --dataset-root data/landcover_ai --output-dir data/landcover_ai_patches --manifest-output data/landcover_ai_patches/manifest.json
