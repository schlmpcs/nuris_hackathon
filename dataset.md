# Dataset Shortlist for NURIS Satellite Extraction

## Task Fit

The target in this repo is segmentation-first extraction of:

- `buildings`
- `roads`
- `water`

with GIS-ready outputs such as `GeoJSON`, plus later vectorization and QGIS validation.

## Best Overall Choices

### 1. LandCover.ai

Best single starting dataset if we want one source that already covers all three target classes.

- Classes include `building`, `water`, and `road`
- RGB orthophotos
- 25 cm and 50 cm resolution
- GeoTIFF rasters and masks
- Good fit for semantic segmentation and later vectorization

Why it fits:
This is the cleanest direct match to the NURIS scope because it already contains the exact core classes we need in a segmentation format.

Link:
https://huggingface.co/datasets/dragon7/LandCover.ai

### 2. LoveDA

Strong multiclass remote-sensing segmentation dataset for general pretraining.

- Includes `building`, `road`, and `water`
- 0.3 m spatial resolution
- Larger and more diverse than LandCover.ai
- Useful for improving generalization across urban and rural environments

Why it fits:
Good candidate for pretraining or mixed training when we want more variety than LandCover.ai alone.

Link:
https://zenodo.org/records/5706578

### 3. SpaceNet

Best official high-quality source when splitting the problem by class.

- Strong building footprint datasets
- Strong road network datasets
- Geospatially rigorous labels
- Widely used in remote sensing benchmarks

Why it fits:
Excellent source for buildings and roads, especially when output quality in GIS matters more than having one all-in-one dataset.

Link:
https://spacenet.ai/datasets/

## Strong Class-Specific Sources

### Buildings

#### HOTOSM VHR Building Segmentation

- Building footprint extraction from aerial imagery
- Hosted on Hugging Face
- Uses validated HOT Tasking Manager projects
- Includes imagery tiles, masks, and GeoJSON labels

Why it fits:
Good recent building-specific dataset with strong annotation provenance.

Link:
https://huggingface.co/datasets/hotosm/vhr-building-segmentation

#### SpaceNet 2 Building Detection

Why it fits:
Still one of the strongest benchmark-style building footprint sources for geospatial model training and vector output workflows.

Link:
https://spacenet.ai/datasets/

### Roads

#### SpaceNet 3 Road Network Detection

- Official SpaceNet road dataset
- Over 8000 km of roads across four AOIs
- Road centerline labels
- AWS public dataset access

Why it fits:
Very strong if road extraction quality is important, especially if we later want centerlines rather than only raster masks.

Link:
https://spacenet.ai/spacenet-roads-dataset/

#### Global-Scale Road Dataset

- Large road graph extraction dataset
- Hosted on Hugging Face
- Global geographic coverage
- Includes urban, rural, and mountainous regions

Caveat:
The Hugging Face page states this is an unofficial mirror, so provenance and licensing should be verified before depending on it.

Why it fits:
Useful if roads are the hardest class and we need more scale and diversity than SpaceNet alone.

Link:
https://huggingface.co/datasets/gaetanbahl/Global-Scale-Road-Dataset

### Water

#### S2 Water Dataset

- Water-only dataset
- Sentinel-2 multispectral imagery
- Includes NIR and SWIR bands
- Binary water masks

Why it fits:
Strong option for water extraction if we want spectral information instead of relying only on RGB imagery.

Link:
https://huggingface.co/datasets/giswqs/s2-water-dataset

## Roboflow Options

These are usable for quick experiments, but they are weaker primary sources than the official or better-documented geospatial datasets above.

### Segmentation Mask Satelite Image

- Classes include `Water`, `Building`, and `Road`
- Semantic segmentation format

Caveat:
Small and lightly documented, so better used for prototyping or augmentation than as the main training source.

Link:
https://universe.roboflow.com/dut-t1sja/segmentation-mask-satelite-image

### Water Segmentation

- Large water-only Roboflow dataset

Caveat:
Useful for quick experiments, but the provenance is less rigorous than the Hugging Face and official geospatial datasets listed above.

Link:
https://universe.roboflow.com/watersegmentation/water-segmentation-n6ecd

## Recommendation

### Simplest path

Start with:

1. `LandCover.ai`
2. `LoveDA`

Reason:
They are the most direct segmentation datasets covering the main classes needed by the NURIS prototype.

### Higher-quality class-by-class path

Use:

1. Buildings: `HOTOSM VHR Building Segmentation` or `SpaceNet`
2. Roads: `SpaceNet 3` or `Global-Scale Road Dataset`
3. Water: `S2 Water Dataset`

Reason:
This should produce stronger per-class performance and better downstream GIS vector outputs, at the cost of more dataset harmonization work.

## Practical Conclusion

If the goal is to move fast on a first prototype, `LandCover.ai` is the best first dataset to try.

If the goal is to maximize quality, use a hybrid training strategy:

- `LandCover.ai` or `LoveDA` for general multiclass pretraining
- `HOTOSM` or `SpaceNet` for buildings
- `SpaceNet 3` or `Global-Scale Road Dataset` for roads
- `S2 Water Dataset` for water

## Sources

- LandCover.ai: https://huggingface.co/datasets/dragon7/LandCover.ai
- LoveDA: https://zenodo.org/records/5706578
- SpaceNet datasets: https://spacenet.ai/datasets/
- SpaceNet 3 roads: https://spacenet.ai/spacenet-roads-dataset/
- HOTOSM buildings: https://huggingface.co/datasets/hotosm/vhr-building-segmentation
- Global-Scale roads: https://huggingface.co/datasets/gaetanbahl/Global-Scale-Road-Dataset
- S2 water: https://huggingface.co/datasets/giswqs/s2-water-dataset
- Roboflow satellite segmentation: https://universe.roboflow.com/dut-t1sja/segmentation-mask-satelite-image
- Roboflow water segmentation: https://universe.roboflow.com/watersegmentation/water-segmentation-n6ecd
