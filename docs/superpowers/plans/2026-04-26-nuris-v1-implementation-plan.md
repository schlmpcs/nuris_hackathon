# NURIS v1 Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version NURIS prototype that ingests provided satellite imagery and an AOI, extracts buildings, roads, and water, and exports GIS-ready GeoJSON plus zone-level indicators.

**Architecture:** The system uses a segmentation-first geospatial pipeline. Raster data is validated and clipped to the AOI, processed in overlapping tiles, passed through pretrained segmentation models, vectorized into buildings, roads, and water features, then merged and cleaned in GIS space before export and reporting.

**Tech Stack:** Python 3.11, GDAL, rasterio, geopandas, shapely, pyproj, numpy, pandas, scikit-image, PyTorch, segmentation-models-pytorch or Hugging Face Transformers, OpenCV, Docker, QGIS

---

## File Structure

### Create

- `README.md`
- `docker/Dockerfile`
- `docker/entrypoint.sh`
- `requirements.txt`
- `src/nuris_pipeline/__init__.py`
- `src/nuris_pipeline/config.py`
- `src/nuris_pipeline/cli.py`
- `src/nuris_pipeline/logging_utils.py`
- `src/nuris_pipeline/io/manifest.py`
- `src/nuris_pipeline/io/raster_loader.py`
- `src/nuris_pipeline/io/vector_loader.py`
- `src/nuris_pipeline/preprocess/crs.py`
- `src/nuris_pipeline/preprocess/aoi.py`
- `src/nuris_pipeline/preprocess/tiling.py`
- `src/nuris_pipeline/models/model_registry.py`
- `src/nuris_pipeline/models/inference.py`
- `src/nuris_pipeline/postprocess/masks.py`
- `src/nuris_pipeline/postprocess/vectorize.py`
- `src/nuris_pipeline/postprocess/merge.py`
- `src/nuris_pipeline/postprocess/filtering.py`
- `src/nuris_pipeline/export/geojson_writer.py`
- `src/nuris_pipeline/export/stats_writer.py`
- `src/nuris_pipeline/qa/control_sample.py`
- `src/nuris_pipeline/qa/metrics.py`
- `configs/base.yaml`
- `configs/v1_inference.yaml`
- `scripts/run_inference.ps1`
- `scripts/run_docker.ps1`
- `qgis/nuris_v1_template.qgz`
- `tests/test_config.py`
- `tests/test_crs.py`
- `tests/test_aoi.py`
- `tests/test_tiling.py`
- `tests/test_vectorize.py`
- `tests/test_merge.py`
- `tests/test_filtering.py`
- `tests/test_stats.py`

### Modify

- none yet; this is a greenfield v1 workspace

## Phase 0: Workspace Bootstrap

### Task 1: Create the repository skeleton

**Files:**
- Create: all directories listed above except the QGIS binary template if unavailable initially

- [ ] **Step 1: Create the directory structure**

Create these folders:

```text
docs/superpowers/specs
docs/superpowers/plans
docker
src/nuris_pipeline/io
src/nuris_pipeline/preprocess
src/nuris_pipeline/models
src/nuris_pipeline/postprocess
src/nuris_pipeline/export
src/nuris_pipeline/qa
configs
scripts
tests
qgis
```

- [ ] **Step 2: Add package markers**

Create:

```python
# src/nuris_pipeline/__init__.py
__all__ = ["__version__"]
__version__ = "0.1.0"
```

- [ ] **Step 3: Add the base README**

Document:

- project purpose
- supported inputs
- supported outputs
- quick-start commands
- known limitations for v1

- [ ] **Step 4: Initialize git if desired**

Run:

```powershell
git init
```

Expected:

- repository initialized successfully

If git is intentionally not desired, skip this step and note it in the report.

## Phase 1: Configuration and CLI

### Task 2: Define runtime configuration

**Files:**
- Create: `src/nuris_pipeline/config.py`
- Create: `configs/base.yaml`
- Create: `configs/v1_inference.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration test**

Test behaviors:

- config file loads from YAML
- required paths are validated
- default CRS export and tiling values are populated

Core assertions:

```python
def test_config_loads_defaults():
    cfg = load_config("configs/base.yaml")
    assert cfg.tiling.tile_size == 1024
    assert cfg.tiling.overlap == 128
    assert cfg.export.geojson_epsg == 4326
```

- [ ] **Step 2: Run the test to confirm failure**

Run:

```powershell
pytest tests/test_config.py -v
```

Expected:

- fail because `load_config` is not implemented

- [ ] **Step 3: Implement configuration loading**

Implement:

- typed config objects with `dataclasses` or `pydantic`
- nested sections for input, CRS, tiling, models, filtering, export, QA
- path normalization and basic validation

- [ ] **Step 4: Add baseline YAML configs**

`configs/base.yaml` should define:

- `tile_size: 1024`
- `overlap: 128`
- `working_crs_strategy: auto_utm`
- `geojson_epsg: 4326`
- thresholds for building, road, water

`configs/v1_inference.yaml` should override:

- model names or checkpoints
- class thresholds
- output paths

- [ ] **Step 5: Re-run tests**

Run:

```powershell
pytest tests/test_config.py -v
```

Expected:

- PASS

### Task 3: Build the command-line entrypoint

**Files:**
- Create: `src/nuris_pipeline/cli.py`
- Create: `src/nuris_pipeline/logging_utils.py`

- [ ] **Step 1: Implement CLI commands**

Minimum commands:

- `validate-inputs`
- `run-inference`
- `build-control-sample`
- `compute-metrics`

- [ ] **Step 2: Add structured logging**

Log:

- input scene ids
- AOI stats
- tile counts
- inference timings
- output paths

- [ ] **Step 3: Smoke test CLI help**

Run:

```powershell
python -m nuris_pipeline.cli --help
```

Expected:

- command list renders without import failures

## Phase 2: Data Ingestion and CRS Handling

### Task 4: Implement raster and vector readers

**Files:**
- Create: `src/nuris_pipeline/io/raster_loader.py`
- Create: `src/nuris_pipeline/io/vector_loader.py`
- Create: `src/nuris_pipeline/io/manifest.py`

- [ ] **Step 1: Add raster metadata extraction**

Return:

- width
- height
- count
- dtype
- nodata
- CRS
- transform
- bounds
- resolution

- [ ] **Step 2: Add AOI and zone loading**

Support:

- GeoJSON
- GeoPackage
- Shapefile

Require:

- polygon geometry for AOI

- [ ] **Step 3: Build a run manifest serializer**

Persist:

- run timestamp
- source scenes
- AOI path
- zone path
- CRS choices
- model identifiers

### Task 5: Implement CRS normalization and validation

**Files:**
- Create: `src/nuris_pipeline/preprocess/crs.py`
- Test: `tests/test_crs.py`

- [ ] **Step 1: Write the failing CRS tests**

Test behaviors:

- AOI is reprojected to the raster CRS
- working CRS auto-selects an appropriate UTM EPSG
- metric calculations reject geographic CRS for area and length

- [ ] **Step 2: Implement CRS utilities**

Functions needed:

- `ensure_crs`
- `reproject_gdf`
- `choose_working_crs`
- `export_to_geojson_crs`

- [ ] **Step 3: Handle missing georeferencing explicitly**

Implement a guard that:

- reads GCP/RPC metadata if present
- otherwise raises a structured exception instructing the operator to georeference before inference

- [ ] **Step 4: Re-run CRS tests**

Run:

```powershell
pytest tests/test_crs.py -v
```

Expected:

- PASS

### Task 6: AOI clipping and masking

**Files:**
- Create: `src/nuris_pipeline/preprocess/aoi.py`
- Test: `tests/test_aoi.py`

- [ ] **Step 1: Write the failing AOI tests**

Test:

- raster is clipped to AOI extent
- mask excludes pixels outside AOI
- empty intersection raises a clear error

- [ ] **Step 2: Implement AOI clip logic**

Implement:

- AOI reprojection to raster CRS
- raster masking using `rasterio.mask`
- output metadata update for clipped imagery

- [ ] **Step 3: Re-run AOI tests**

Run:

```powershell
pytest tests/test_aoi.py -v
```

Expected:

- PASS

## Phase 3: Tiling

### Task 7: Tile generation with overlap and core windows

**Files:**
- Create: `src/nuris_pipeline/preprocess/tiling.py`
- Test: `tests/test_tiling.py`

- [ ] **Step 1: Write the failing tiling tests**

Test:

- image is split into expected tile count
- edge tiles clamp correctly
- each tile records both full and core windows

- [ ] **Step 2: Implement tile generation**

Implement data structures for:

- tile id
- raster window
- core window
- affine transform
- bounds in working CRS

- [ ] **Step 3: Add tile export helpers**

Allow:

- lazy reading from source raster
- optional cached tile writes for debugging

- [ ] **Step 4: Re-run tiling tests**

Run:

```powershell
pytest tests/test_tiling.py -v
```

Expected:

- PASS

## Phase 4: Model Inference

### Task 8: Model registry and checkpoint abstraction

**Files:**
- Create: `src/nuris_pipeline/models/model_registry.py`
- Create: `src/nuris_pipeline/models/inference.py`

- [ ] **Step 1: Define a model interface**

Each model wrapper should provide:

- `load()`
- `predict(tile_array)`
- `class_names`
- `probability_maps`

- [ ] **Step 2: Implement a v1 pretrained registry**

Support:

- segmentation backbone name
- checkpoint path or Hugging Face identifier
- device selection
- inference tile normalization

- [ ] **Step 3: Add batch inference orchestration**

Pipeline behavior:

- read tile data
- normalize bands
- run model
- write class probability rasters or in-memory arrays

### Task 9: Thresholding and mask preparation

**Files:**
- Create: `src/nuris_pipeline/postprocess/masks.py`

- [ ] **Step 1: Implement per-class thresholding**

Support:

- class-specific thresholds
- optional morphological open/close
- nodata masking

- [ ] **Step 2: Add threshold diagnostics**

Persist optional previews:

- raw probability heatmaps
- binary masks
- confidence histograms

## Phase 5: Vectorization and Merge

### Task 10: Vectorize buildings, roads, and water

**Files:**
- Create: `src/nuris_pipeline/postprocess/vectorize.py`
- Test: `tests/test_vectorize.py`

- [ ] **Step 1: Write failing vectorization tests**

Test:

- polygon masks become valid polygons
- road masks become lines
- attributes include provisional class and tile source

- [ ] **Step 2: Implement building and water polygonization**

Use:

- connected components
- polygon extraction
- validity repair

- [ ] **Step 3: Implement road line extraction**

Use:

- morphology cleanup
- skeletonization
- line tracing

- [ ] **Step 4: Re-run vectorization tests**

Run:

```powershell
pytest tests/test_vectorize.py -v
```

Expected:

- PASS

### Task 11: Merge across overlapping tiles

**Files:**
- Create: `src/nuris_pipeline/postprocess/merge.py`
- Test: `tests/test_merge.py`

- [ ] **Step 1: Write failing merge tests**

Test:

- duplicate polygons from overlap are merged
- road segments crossing tiles are snapped and dissolved
- centroid or representative point core-window retention works

- [ ] **Step 2: Implement merge logic**

Shared merge behavior:

- convert all features into working CRS
- core-window retention
- class-wise merge passes

Building/water:

- dissolve by overlap and proximity

Road:

- endpoint snapping
- segment merge

- [ ] **Step 3: Re-run merge tests**

Run:

```powershell
pytest tests/test_merge.py -v
```

Expected:

- PASS

## Phase 6: Filtering and Confidence

### Task 12: Apply class-specific filtering rules

**Files:**
- Create: `src/nuris_pipeline/postprocess/filtering.py`
- Test: `tests/test_filtering.py`

- [ ] **Step 1: Write failing filtering tests**

Test:

- tiny buildings are removed
- short road stubs are removed
- small speckle water polygons are removed
- invalid geometries are repaired or dropped

- [ ] **Step 2: Implement filtering**

Rules should include:

- minimum area for buildings
- minimum area for water
- minimum length for roads
- compactness or elongation sanity checks
- duplicate removal by IoU or near-equality

- [ ] **Step 3: Add confidence computation**

Implement:

- object-level score from mean probability
- overlap consistency bonus
- geometry plausibility penalty

- [ ] **Step 4: Re-run filtering tests**

Run:

```powershell
pytest tests/test_filtering.py -v
```

Expected:

- PASS

## Phase 7: Export and Reporting

### Task 13: GeoJSON export

**Files:**
- Create: `src/nuris_pipeline/export/geojson_writer.py`

- [ ] **Step 1: Implement export schema enforcement**

Required properties:

- `id`
- `class`
- `confidence`
- `source_id`

Optional:

- `area_m2`
- `length_m`
- `zone_id`
- `acq_date`

- [ ] **Step 2: Reproject to GeoJSON CRS**

Behavior:

- export in `EPSG:4326`
- preserve metric fields computed in working CRS

- [ ] **Step 3: Validate output in a GIS-friendly way**

Checks:

- no missing geometry
- valid GeoJSON structure
- only supported geometry types per class

### Task 14: Zone statistics

**Files:**
- Create: `src/nuris_pipeline/export/stats_writer.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write failing stats tests**

Test:

- counts by class are correct
- density per square kilometer is computed from zone area
- building and water area sum correctly
- road length sum correctly

- [ ] **Step 2: Implement zone aggregation**

Output tables:

- per-zone metrics
- optional per-class summary

- [ ] **Step 3: Re-run stats tests**

Run:

```powershell
pytest tests/test_stats.py -v
```

Expected:

- PASS

## Phase 8: QA Workflow

### Task 15: Manual control sample builder

**Files:**
- Create: `src/nuris_pipeline/qa/control_sample.py`

- [ ] **Step 1: Implement sample generation**

Sample dimensions:

- by zone
- by confidence bin
- by land-cover context if metadata is available

- [ ] **Step 2: Export review packages**

Produce:

- sampled tile list
- candidate feature list
- review GeoPackage or GeoJSON for QGIS

### Task 16: QA metrics computation

**Files:**
- Create: `src/nuris_pipeline/qa/metrics.py`

- [ ] **Step 1: Implement metrics**

Support:

- Precision
- Recall
- F1
- IoU for polygons
- buffered overlap metric for roads

- [ ] **Step 2: Add per-class reporting**

Output:

- class-level summary
- global summary
- confusion notes template for the report

## Phase 9: Packaging and Delivery

### Task 17: Dockerize the pipeline

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/entrypoint.sh`
- Create: `requirements.txt`
- Create: `scripts/run_docker.ps1`

- [ ] **Step 1: Define Python and system dependencies**

Must include:

- GDAL-compatible runtime
- geospatial Python libraries
- model framework

- [ ] **Step 2: Add a reproducible container entrypoint**

Entrypoint should support:

- config-driven run
- mounted input and output directories

- [ ] **Step 3: Smoke test the container**

Run:

```powershell
docker build -f docker/Dockerfile -t nuris-v1 .
```

Expected:

- successful image build

### Task 18: Add operator scripts and documentation

**Files:**
- Create: `scripts/run_inference.ps1`
- Create: `README.md`

- [ ] **Step 1: Add local run script**

Script should:

- validate environment
- run input validation
- invoke pipeline config

- [ ] **Step 2: Document operator workflow**

README sections:

- prerequisites
- input layout
- example commands
- output structure
- QA workflow
- known limitations

### Task 19: QGIS project and delivery bundle

**Files:**
- Create: `qgis/nuris_v1_template.qgz`

- [ ] **Step 1: Define QGIS layer expectations**

Bundle should include:

- basemap instructions
- AOI layer
- detections layer
- zones layer
- style categories for classes

- [ ] **Step 2: Prepare delivery folder conventions**

Expected output structure:

```text
outputs/
  run_YYYYMMDD_HHMMSS/
    vectors/
    rasters/
    qa/
    stats/
    logs/
```

## Verification Gates

### Task 20: End-to-end integration checks

**Files:**
- No new files required

- [ ] **Step 1: Run unit tests**

Run:

```powershell
pytest tests -v
```

Expected:

- all implemented tests pass

- [ ] **Step 2: Run a dry validation on a sample scene**

Run:

```powershell
python -m nuris_pipeline.cli validate-inputs --config configs/v1_inference.yaml
```

Expected:

- input validation summary with CRS, AOI overlap, and tile count

- [ ] **Step 3: Run one inference sample**

Run:

```powershell
python -m nuris_pipeline.cli run-inference --config configs/v1_inference.yaml
```

Expected:

- GeoJSON written successfully
- zone stats written successfully
- logs include run manifest and timing summary

- [ ] **Step 4: Open results in QGIS**

Verify:

- alignment with basemap
- correct class styling
- no obvious tile seam duplication

## Execution Order Recommendation

1. Phase 0 and Phase 1
2. Phase 2 and Phase 3
3. Phase 4 and Phase 5
4. Phase 6 and Phase 7
5. Phase 8 and Phase 9
6. Verification Gates

## v1 Tradeoffs to Preserve

- Prefer reliable major-road extraction over aggressive small-road detection.
- Fail loudly on missing georeferencing rather than silently producing shifted outputs.
- Keep confidence conservative when calibration data is weak.
- Prioritize valid GIS output and reproducibility over model complexity.

## Notes

- The current workspace is not yet a git repository. If version control is required for the deliverable workflow, initialize git before implementation begins.
- If imagery resolution turns out to be too coarse for roads, the road class should be downgraded to major roads only and that decision should be reflected in the report and thresholds.
