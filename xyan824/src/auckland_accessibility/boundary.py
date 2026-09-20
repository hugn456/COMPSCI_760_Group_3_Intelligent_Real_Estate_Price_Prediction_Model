from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd

from .http import HttpClient
from .manifest import record_artifact, utc_now
from .paths import PROCESSED_DIR, RAW_DIR, load_json_config


RAW_BOUNDARY = RAW_DIR / "boundary" / "territorial_authority_2026_auckland.geojson"
PROCESSED_BOUNDARY = PROCESSED_DIR / "auckland_boundary.geojson"


def _find_name_field(metadata: dict[str, Any]) -> str:
    fields = metadata.get("fields", [])
    candidates: list[str] = []
    for field in fields:
        name = str(field.get("name", ""))
        alias = str(field.get("alias", ""))
        if "name" in name.lower() or "name" in alias.lower():
            candidates.append(name)
    preferred = [name for name in candidates if "ascii" not in name.lower()]
    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]
    raise RuntimeError("Could not identify a name field in the Stats NZ boundary layer")


def collect_boundary(client: HttpClient, *, refresh: bool = False) -> Path:
    cfg = load_json_config("sources.json")["boundary"]
    layer_url = cfg["layer_url"]
    boundary_name = cfg["name"]
    RAW_BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_BOUNDARY.parent.mkdir(parents=True, exist_ok=True)

    if RAW_BOUNDARY.exists() and not refresh:
        payload = json.loads(RAW_BOUNDARY.read_text(encoding="utf-8"))
    else:
        metadata = client.get_json(layer_url, params={"f": "json"})
        name_field = _find_name_field(metadata)
        payload = client.get_json(
            f"{layer_url}/query",
            params={
                "where": f"{name_field}='{boundary_name}'",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
            },
        )
        if payload.get("type") != "FeatureCollection":
            raise RuntimeError(f"Boundary endpoint did not return GeoJSON: {payload}")
        tmp = RAW_BOUNDARY.with_suffix(".geojson.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(RAW_BOUNDARY)

    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if gdf.empty:
        raise RuntimeError("Stats NZ boundary query returned no Auckland geometry")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf.to_file(PROCESSED_BOUNDARY, driver="GeoJSON")
    record_artifact(
        "boundary.raw",
        RAW_BOUNDARY,
        source_url=layer_url,
        retrieved_at=utc_now(),
        feature_count=len(gdf),
        license="Stats NZ Crown copyright; see source metadata",
    )
    record_artifact(
        "boundary.processed",
        PROCESSED_BOUNDARY,
        feature_count=len(gdf),
        crs="EPSG:4326",
    )
    return PROCESSED_BOUNDARY


def load_boundary(path: Path = PROCESSED_BOUNDARY) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Boundary not found: {path}. Run the boundary collector first.")
    return gpd.read_file(path).to_crs("EPSG:4326")

