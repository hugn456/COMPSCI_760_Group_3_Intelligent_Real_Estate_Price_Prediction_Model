"""OSMnx network acquisition, using its official routing/save-load examples.

References and licenses are recorded in TRAVEL_DISTANCE_SOURCES.md.
Only public study-area polygons are sent to Overpass, never property rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

from .boundary import load_boundary
from .manifest import sha256_file, utc_now
from .paths import RAW_DIR, ROOT


def study_area(config: dict, scope: str, buffer_m: float | None = None):
    if scope == "pilot":
        core = gpd.GeoSeries([box(*config["pilot_bbox_wsen"])], crs=4326).to_crs(2193)
    elif scope == "full":
        core = gpd.GeoSeries([load_boundary().geometry.union_all()], crs=4326).to_crs(2193)
    else:
        raise ValueError(f"Unknown scope: {scope}")
    buffer = config["buffer_m"] if buffer_m is None else buffer_m
    polygon = core.buffer(buffer).to_crs(4326).iloc[0]
    return core.iloc[0], polygon


def graph_paths(config: dict, mode: str, polygon):
    if mode not in {"walk", "drive"}:
        raise ValueError(mode)
    signature = {
        "mode": mode, "polygon_wkb": polygon.wkb_hex, "osmnx": ox.__version__,
        "simplify": config["simplify"], "retain_all": True,
        "endpoint": config["overpass_url"], "protocol": 1,
    }
    if config.get("network_source") == "pbf":
        pbf = (ROOT / config["pbf_path"]).resolve()
        if not pbf.exists():
            raise FileNotFoundError(f"Download the public Geofabrik NZ snapshot first: {pbf}")
        signature["pbf_sha256"] = sha256_file(pbf)
        signature["network_source"] = "pbf"
        signature["pbf_extract_protocol"] = 1
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    folder = RAW_DIR / "travel_network"
    folder.mkdir(parents=True, exist_ok=True)
    if config.get("compact_network"):
        return folder / f"{mode}_{key}.npz", folder / f"{mode}_{key}.npz.json", signature
    return folder / f"{mode}_{key}.graphml", folder / f"{mode}_{key}.json", signature


def download_graph(config: dict, mode: str, polygon) -> Path:
    destination, meta_path, signature = graph_paths(config, mode, polygon)
    if destination.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if sha256_file(destination) == meta["sha256"]:
            print(f"CACHE {mode}: {destination.name}", flush=True)
            return destination
        raise ValueError(f"Graph cache hash mismatch: {destination}")
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(RAW_DIR / "travel_network" / "http_cache")
    ox.settings.log_console = True
    ox.settings.requests_timeout = config["request_timeout_seconds"]
    ox.settings.overpass_url = config["overpass_url"]
    ox.settings.http_user_agent = "COMPSCI760-Auckland-accessibility/1.0 (OSMnx research)"
    started = time.perf_counter()
    if config.get("network_source") == "pbf":
        from .travel_pbf import extract_mode_xml
        xml_path = destination.with_suffix(".osm")
        extract_mode_xml((ROOT / config["pbf_path"]).resolve(), xml_path, mode, polygon)
        if config.get("compact_network"):
            from .travel_compact import compact_from_xml
            if config["simplify"]:
                raise ValueError("Compact backend preserves unsimplified nodes only")
            graph = compact_from_xml(xml_path, mode)
        else:
            graph = ox.graph.graph_from_xml(xml_path, bidirectional=(mode == "walk"),
                                           simplify=config["simplify"], retain_all=True)
            graph = ox.truncate.truncate_graph_polygon(graph, polygon, truncate_by_edge=True)
    else:
        graph = ox.graph.graph_from_polygon(
            polygon, network_type=mode, simplify=config["simplify"], retain_all=True,
            truncate_by_edge=True,
        )
    temp = destination.with_suffix(".tmp" + destination.suffix)
    if config.get("compact_network"):
        graph.save(temp)
    else:
        ox.io.save_graphml(graph, filepath=temp)
    temp.replace(destination)
    meta = {"signature": signature, "created_at": utc_now(), "sha256": sha256_file(destination),
            "nodes": len(graph), "edges": graph.number_of_edges(),
            "seconds": round(time.perf_counter() - started, 2),
            "storage": "compact_npz" if config.get("compact_network") else "graphml",
            "attribution": "OpenStreetMap contributors, ODbL 1.0"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def ensure_graph(config_path: Path, mode: str, scope: str, buffer_m=None) -> tuple[Path, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, polygon = study_area(config, scope, buffer_m)
    path, metadata, _ = graph_paths(config, mode, polygon)
    command = [sys.executable, "-m", "auckland_accessibility.travel_network", "--config", str(config_path),
               "--mode", mode, "--scope", scope]
    if buffer_m is not None:
        command += ["--buffer-m", str(buffer_m)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    # OSMnx may retry HTTP errors internally: bound the entire child process.
    subprocess.run(command, env=env, check=True, timeout=config["graph_wall_timeout_seconds"])
    return path, json.loads(metadata.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["walk", "drive"], required=True)
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--buffer-m", type=float)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    _, shape = study_area(cfg, args.scope, args.buffer_m)
    download_graph(cfg, args.mode, shape)
