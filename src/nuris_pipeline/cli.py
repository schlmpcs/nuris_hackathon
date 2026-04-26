from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from nuris_pipeline.config import load_config
from nuris_pipeline.data.landcover_ai import (
    build_landcover_patch_manifest,
    discover_landcover_ai_samples,
    get_landcover_ai_download_spec,
    prepare_landcover_ai_patches,
    write_landcover_ai_manifest,
    write_landcover_patch_manifest,
)
from nuris_pipeline.export.geojson_writer import write_geojson
from nuris_pipeline.export.stats_writer import summarize_by_zone
from nuris_pipeline.io.manifest import SceneManifest, build_run_manifest, write_manifest
from nuris_pipeline.io.raster_loader import read_raster_metadata
from nuris_pipeline.io.vector_loader import load_polygon_layer
from nuris_pipeline.logging_utils import configure_logging
from nuris_pipeline.models.inference import run_model_inference
from nuris_pipeline.models.model_registry import load_model
from nuris_pipeline.postprocess.filtering import filter_features
from nuris_pipeline.postprocess.masks import threshold_probability_map
from nuris_pipeline.postprocess.merge import merge_polygon_features, merge_road_features
from nuris_pipeline.postprocess.vectorize import vectorize_buildings, vectorize_roads, vectorize_water
from nuris_pipeline.preprocess.crs import choose_working_crs, reproject_gdf
from nuris_pipeline.preprocess.tiling import generate_tiles
from nuris_pipeline.qa.control_sample import build_control_sample
from nuris_pipeline.qa.metrics import compute_detection_metrics


LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nuris-pipeline")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("validate-inputs", "run-inference", "build-control-sample", "compute-metrics"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=name != "compute-metrics")
        if name == "compute-metrics":
            sub.add_argument("--predicted", required=True)
            sub.add_argument("--truth", required=True)
            sub.add_argument("--output", required=True)
        if name == "build-control-sample":
            sub.add_argument("--features", required=True)
            sub.add_argument("--output", required=True)

    landcover = subparsers.add_parser("prepare-landcover-ai")
    landcover.add_argument("--dataset-root", required=True)
    landcover.add_argument("--output", required=True)

    landcover_download = subparsers.add_parser("download-landcover-ai")
    landcover_download.add_argument("--output-dir", required=True)

    landcover_patches = subparsers.add_parser("prepare-landcover-ai-patches")
    landcover_patches.add_argument("--dataset-root", required=True)
    landcover_patches.add_argument("--output-dir", required=True)
    landcover_patches.add_argument("--manifest-output", required=True)

    training = subparsers.add_parser("train-segmentation")
    training.add_argument("--config", required=True)

    return parser


def _scene_manifests(scene_paths: list[str]) -> list[SceneManifest]:
    manifests: list[SceneManifest] = []
    for scene_path in scene_paths:
        metadata = read_raster_metadata(scene_path)
        manifests.append(
            SceneManifest(
                source_id=Path(scene_path).stem,
                path=str(scene_path),
                crs=metadata.crs or "UNKNOWN",
                width=metadata.width,
                height=metadata.height,
                resolution_x=float(metadata.resolution[0]),
                resolution_y=float(metadata.resolution[1]),
            )
        )
    return manifests


def validate_inputs(config_path: str) -> int:
    cfg = load_config(config_path)
    aoi = load_polygon_layer(cfg.input.aoi_path)
    scenes = _scene_manifests(cfg.input.scene_paths)

    working_crs = choose_working_crs(aoi)
    LOGGER.info("Validated %s scene(s); working CRS=%s", len(scenes), working_crs.to_string())

    return 0


def run_inference(config_path: str) -> int:
    cfg = load_config(config_path)
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = load_polygon_layer(cfg.input.aoi_path)
    zones = load_polygon_layer(cfg.input.zones_path) if cfg.input.zones_path else aoi.assign(zone_id="aoi")
    working_crs = choose_working_crs(aoi)
    aoi_working = reproject_gdf(aoi, working_crs)
    zones_working = reproject_gdf(zones, working_crs)

    model = load_model(cfg.model.backend, cfg.model.checkpoint, cfg.model.device)
    feature_frames: list[gpd.GeoDataFrame] = []
    scenes = []

    for scene_path in cfg.input.scene_paths:
        metadata = read_raster_metadata(scene_path)
        scenes.append(
            SceneManifest(
                source_id=Path(scene_path).stem,
                path=str(scene_path),
                crs=metadata.crs or "UNKNOWN",
                width=metadata.width,
                height=metadata.height,
                resolution_x=float(metadata.resolution[0]),
                resolution_y=float(metadata.resolution[1]),
            )
        )

        with rasterio.open(scene_path) as src:
            tiles = generate_tiles(src.width, src.height, cfg.tiling.tile_size, cfg.tiling.overlap, src.transform)
            for tile in tiles:
                tile_array = src.read(window=tile.window)
                result = run_model_inference(model, tile_array, Path(scene_path).stem, tile.tile_id)

                building_mask = threshold_probability_map(result.probability_maps["building"], cfg.model.class_thresholds["building"])
                road_mask = threshold_probability_map(result.probability_maps["road"], cfg.model.class_thresholds["road"])
                water_mask = threshold_probability_map(result.probability_maps["water"], cfg.model.class_thresholds["water"])

                building_gdf = vectorize_buildings(building_mask, tile.transform, result.source_id, result.tile_id)
                road_gdf = vectorize_roads(road_mask, tile.transform, result.source_id, result.tile_id)
                water_gdf = vectorize_water(water_mask, tile.transform, result.source_id, result.tile_id)

                for frame in (building_gdf, road_gdf, water_gdf):
                    if not frame.empty:
                        frame = frame.set_crs(src.crs)
                        frame = frame.to_crs(working_crs)
                        feature_frames.append(frame)

    if not feature_frames:
        LOGGER.warning("No features produced")
        return 0

    features = gpd.GeoDataFrame(pd.concat(feature_frames, ignore_index=True), geometry="geometry", crs=working_crs)
    merged_buildings = merge_polygon_features(features[features["class"] == "building"], "building")
    merged_roads = merge_road_features(features[features["class"] == "road"])
    merged_water = merge_polygon_features(features[features["class"] == "water"], "water")
    merged = gpd.GeoDataFrame(pd.concat([merged_buildings, merged_roads, merged_water], ignore_index=True), geometry="geometry", crs=working_crs)

    filtered = filter_features(
        merged,
        building_min_area_m2=cfg.filtering.building_min_area_m2,
        road_min_length_m=cfg.filtering.road_min_length_m,
        water_min_area_m2=cfg.filtering.water_min_area_m2,
    )

    filtered = filtered.reset_index(drop=True)
    filtered["id"] = [f"obj_{index:06d}" for index in range(1, len(filtered) + 1)]
    filtered["area_m2"] = filtered.geometry.area.where(filtered.geometry.geom_type.isin(["Polygon", "MultiPolygon"]))
    filtered["length_m"] = filtered.geometry.length.where(filtered.geometry.geom_type.isin(["LineString", "MultiLineString"]))
    filtered["source_id"] = filtered["source_id"].fillna("unknown")

    vector_output = output_dir / "detections.geojson"
    write_geojson(filtered, vector_output, epsg=cfg.export.geojson_epsg)

    stats = summarize_by_zone(filtered, zones_working)
    stats_output = output_dir / "zone_stats.csv"
    stats.to_csv(stats_output, index=False)

    manifest = build_run_manifest(config_path, cfg.input.aoi_path, cfg.input.zones_path, scenes)
    write_manifest(manifest, output_dir / "run_manifest.json")
    LOGGER.info("Wrote outputs to %s", output_dir)
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "validate-inputs":
        return validate_inputs(args.config)
    if args.command == "run-inference":
        return run_inference(args.config)
    if args.command == "build-control-sample":
        cfg = load_config(args.config)
        features = gpd.read_file(args.features)
        sample = build_control_sample(features, cfg.qa.sample_size, cfg.qa.confidence_bins)
        write_geojson(sample, args.output, epsg=cfg.export.geojson_epsg)
        return 0
    if args.command == "compute-metrics":
        predicted = gpd.read_file(args.predicted)
        truth = gpd.read_file(args.truth)
        metrics = compute_detection_metrics(predicted, truth)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output, index=False)
        return 0
    if args.command == "prepare-landcover-ai":
        samples = discover_landcover_ai_samples(args.dataset_root)
        write_landcover_ai_manifest(samples, args.output)
        LOGGER.info("Prepared %s LandCover.ai samples", len(samples))
        return 0
    if args.command == "download-landcover-ai":
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "download-landcover-ai requires huggingface_hub. Install it with "
                "'python -m pip install huggingface_hub'."
            ) from exc

        spec = get_landcover_ai_download_spec()
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            allow_patterns=spec["allow_patterns"],
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
        )
        LOGGER.info("Downloaded LandCover.ai to %s", args.output_dir)
        return 0
    if args.command == "prepare-landcover-ai-patches":
        patches = prepare_landcover_ai_patches(args.dataset_root, args.output_dir)
        write_landcover_patch_manifest(args.dataset_root, args.output_dir, args.manifest_output)
        LOGGER.info("Prepared %s LandCover.ai patches", len(patches))
        return 0
    if args.command == "train-segmentation":
        from nuris_pipeline.training.trainer import train_segmentation

        return train_segmentation(args.config)

    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
