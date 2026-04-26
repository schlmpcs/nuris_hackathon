# NURIS Satellite Object Detection v1 Design

**Status:** Approved design baseline  
**Date:** 2026-04-26  
**Project:** NURIS satellite imagery object extraction prototype  
**Source requirement:** `Техническое задание для NURIS (2).pdf`

## Goal

Build a first-version prototype that processes provided satellite imagery inside a supplied AOI, automatically extracts `buildings`, `roads`, and `water`, and delivers GIS-ready outputs that open correctly in QGIS without manual adjustment.

The primary required output is `GeoJSON`. The prototype must also support tiled processing, object-level confidence scoring, post-processing, and quantitative indicators by zone.

## Scope

### In Scope

- GeoTIFF ingestion and validation
- AOI-aware clipping and tile generation
- Inference on provided imagery
- Segmentation-first extraction of:
  - `building` as Polygon
  - `road` as LineString or MultiLineString
  - `water` as Polygon
- Tile result merging and seam handling
- GIS post-processing and geometry cleanup
- GeoJSON export with required attributes
- Zone-level summary statistics
- Manual quality-control workflow when full labels are absent
- Dockerized runnable codebase
- QGIS project for visual verification

### Out of Scope for v1

- Full production orchestration
- Temporal change detection beyond schema placeholders
- Complex active-learning loops
- Large-scale training from scratch
- Multi-sensor fusion beyond optional later extension

## Constraints from the NURIS Technical Specification

- `GeoJSON` is mandatory as the primary result format.
- Outputs must overlay standard base maps in GIS software without manual adjustment.
- The coordinate reference system must be explicit and correct.
- Processing must support tiling with later result assembly.
- Post-processing must reduce false positives through filtering, duplicate removal, and rules.
- Reporting must include quantitative indicators by zone.
- If source data lacks georeferencing, the method and accuracy limits must be documented.

## v1 Operating Assumptions

- Organizers provide at least one georeferenced raster scene and an AOI polygon.
- Imagery resolution is sufficient for extracting buildings, major roads, and water bodies.
- A zone layer may be provided separately; if not, the AOI itself is treated as the default zone.
- v1 is inference-first using pretrained weights, with a later light fine-tuning path if sample labels become available.

## Recommended Architecture

Use a **segmentation-first hybrid pipeline**:

1. Read imagery and AOI.
2. Clip or mask the raster to the AOI.
3. Split the image into overlapping tiles.
4. Run semantic segmentation for `building`, `road`, and `water`.
5. Convert masks into vector geometries by class.
6. Merge tile outputs in map coordinates.
7. Apply GIS cleanup, deduplication, and class-specific rules.
8. Export GeoJSON and summary tables.

This is the best fit for the chosen classes because:

- buildings and water are naturally represented as polygons;
- roads are better extracted as raster masks and converted to centerlines than detected as independent boxed objects;
- tile-based segmentation scales well and keeps geospatial handling simple;
- later fine-tuning can improve one or all classes without changing the rest of the pipeline.

## Data Flow

### 1. Ingestion

Inputs:

- one or more `GeoTIFF` scenes
- AOI polygon
- optional zone polygons
- optional DEM or context layers

Validation checks:

- file readability
- CRS existence
- affine transform validity
- band count and dtype
- pixel resolution
- nodata definition
- AOI intersection with scene footprint

Each input scene is registered in a manifest with:

- `source_id`
- original path
- raster CRS
- resolution
- acquisition date if available
- AOI overlap area

### 2. CRS Strategy

Processing CRS rules:

- Use the raster native CRS if it is projected and appropriate for metric work.
- Otherwise reproject to a project working CRS, preferably the AOI UTM zone.
- Use the working CRS for all area, length, and topological operations.
- Export GeoJSON in `EPSG:4326` for broad GIS compatibility, while optionally also writing a projected `GeoPackage`.

If georeferencing is missing:

1. inspect for embedded `RPC` or `GCP` metadata;
2. if absent, georeference against trusted reference data using manually or semi-automatically selected GCPs;
3. warp to the target EPSG with GDAL;
4. record positional RMSE and include an accuracy limitation note in the report.

### 3. AOI Handling

- Reproject AOI to the raster or working CRS before clipping.
- Use the AOI polygon as a hard spatial mask.
- Do not run inference outside the AOI bounds.
- Persist both the AOI geometry and clipped raster footprint in the run manifest for reproducibility.

### 4. Tiling

Default v1 tile strategy:

- tile size: `1024 x 1024` pixels
- overlap: `128` pixels
- stride: `896` pixels

Tile metadata:

- tile id
- `source_id`
- row and column index
- pixel window
- affine transform
- core window excluding overlap margins

Rationale:

- large enough for context needed by roads;
- small enough for GPU inference on common hardware;
- overlap reduces border artifacts on all three classes.

### 5. Inference

Recommended model family:

- `SegFormer` or `U-Net` class for semantic segmentation

Recommended v1 approach:

- start with a single multi-class segmentation model if pretrained weights are suitable;
- otherwise use one shared backbone with per-class output heads, or independent class models if that reduces integration risk.

Expected outputs per tile:

- per-class probability rasters
- optional binary masks after thresholding
- per-tile metadata and execution timings

### 6. Vectorization

`building`:

- threshold mask
- connected components
- polygonize
- remove tiny fragments

`water`:

- threshold mask
- polygonize
- fill tiny holes
- preserve larger internal holes when hydrologically meaningful

`road`:

- threshold mask
- clean narrow noise
- skeletonize
- convert skeleton to linework
- snap and merge line segments

### 7. Tile Reassembly

- Convert all vectors into working CRS immediately.
- Keep only detections whose representative point falls inside the tile core window.
- Merge seam-crossing polygons by intersection and distance rules.
- Merge road segments by endpoint snapping and line dissolve.

This avoids double counting while preserving detections that span tile boundaries.

### 8. Post-Processing Rules

Shared rules:

- remove invalid geometries
- fix self-intersections
- remove duplicates by IoU or near-identical geometry
- drop detections in nodata areas

Building rules:

- minimum area threshold
- compactness filter
- maximum hole ratio

Road rules:

- minimum length threshold
- minimum average width proxy from raster support
- remove isolated short stubs

Water rules:

- minimum area threshold
- smooth jagged boundaries
- remove implausible speckle polygons

### 9. Confidence Scoring

Each feature receives a `0-100` confidence score.

Object confidence is derived from:

- mean class probability inside the feature
- boundary stability
- agreement across overlapping tiles
- class-specific plausibility checks

Base formula for v1:

`confidence = round(100 * calibrated_score)`

Where `calibrated_score` is a weighted combination of:

- model probability support
- overlap consistency
- geometry plausibility

Later, if a labeled sample becomes available, the score can be calibrated using isotonic regression or temperature scaling.

## Output Schema

Primary output: `GeoJSON`

Each feature must contain:

- `id`
- `class`
- `confidence`
- `source_id`
- geometry

Class-dependent optional attributes:

- `area_m2` for polygons
- `length_m` for roads
- `zone_id`
- `acq_date`
- `notes`

Example schema:

```json
{
  "type": "Feature",
  "properties": {
    "id": "bld_000001",
    "class": "building",
    "confidence": 91,
    "source_id": "scene_01",
    "area_m2": 146.8,
    "length_m": null,
    "zone_id": "zone_a"
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": []
  }
}
```

## Zone-Level Statistics

For each zone:

- count of objects by class
- density per square kilometer
- total building area
- total water area
- total road length

Calculation rules:

- intersect output features with zone polygons in the working CRS;
- compute area and length in metric units;
- aggregate into a flat table suitable for CSV, report tables, and GIS joins.

If no zone layer is supplied:

- use the AOI as a single zone;
- state this assumption in the report.

## Quality Assurance

### If no full ground truth exists

Create a manual control sample:

- stratify across different land-cover contexts;
- include easy, medium, and difficult image regions;
- sample detections across low, medium, and high confidence bins;
- manually annotate truth in QGIS on selected tiles or zones.

Metrics:

- Precision
- Recall
- F1
- IoU for polygons
- length overlap or buffered IoU for roads

### Error Cases to Document

- building shadows merged into structures
- roads confused with bare soil or parking areas
- water confusion with dark roofs or shadowed terrain
- seasonal variation
- mixed land cover
- positional offsets from weak georeferencing
- tile seam artifacts

## Fine-Tuning Path

v1 does not depend on custom training, but the design supports a later light fine-tuning stage:

- collect a small labeled set from representative AOI patches;
- fine-tune the chosen backbone for a few epochs;
- recalibrate thresholds and confidence mapping;
- compare against the zero-shot baseline on the same manual control sample.

## Deliverables

- Dockerized source code with run instructions
- technical report
- GeoJSON results for at least two example territories if data volume permits
- zone statistics table
- QGIS project for verification and screenshots
- optional GeoPackage export

## Risks and Mitigations

- **Weak imagery resolution:** reduce road ambition to major roads only and document limitations.
- **Missing or poor CRS:** isolate georeferencing as a mandatory validation gate before inference.
- **Class confusion:** tighten post-processing and route low-confidence outputs to manual review.
- **GPU limitations:** reduce tile size or batch size while preserving overlap.
- **Sparse evaluation data:** use a stratified manual control protocol and make confidence calibration conservative.

## Acceptance Criteria for v1

- The pipeline runs from input imagery and AOI to GIS outputs without manual geometry editing.
- GeoJSON loads correctly in QGIS and overlays on a basemap without manual shifting.
- The output includes `building`, `road`, and `water` classes with per-feature confidence.
- The run produces a zone summary table.
- The technical report explains CRS handling, methods, metrics, and limitations.
