"""Offline OSM extract via Pyosmium's existing BackReferenceWriter API.

The tag predicate is translated from pinned OSMnx's own mode filter rather
than maintaining a separate list of roads. Unsupported expressions fail closed.
"""
import json
from pathlib import Path
import re
import hashlib
import requests

import osmium
import osmnx as ox
from shapely.geometry import LineString
from shapely.prepared import prep

from .manifest import sha256_file


def download_snapshot(destination: Path):
    """Download once, then verify cached snapshot; never silently refresh."""
    metadata_path = destination.with_suffix(".json")
    if destination.exists():
        meta = json.loads(metadata_path.read_text())
        if sha256_file(destination) != meta["sha256"]:
            raise ValueError("PBF checksum mismatch")
        return meta
    url = "https://download.geofabrik.de/australia-oceania/new-zealand-latest.osm.pbf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    digest = hashlib.sha256()
    size = 0
    with requests.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if "Content-Length" in response.headers and size != int(response.headers["Content-Length"]):
            raise ValueError("Incomplete PBF download")
        meta = {"url": url, "bytes": size, "sha256": digest.hexdigest(),
                "last_modified": response.headers.get("Last-Modified")}
    temporary.replace(destination)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def compile_osmnx_filter(expression):
    token = re.compile(r'\["([^"\]]+)"(?:(!~|~|!=|=)"([^"\]]*)")?\]')
    matches = list(token.finditer(expression))
    if "".join(m.group(0) for m in matches) != expression:
        raise ValueError(f"Unsupported OSMnx filter: {expression}")
    def accepts(tags):
        for match in matches:
            key, operator, value = match.groups()
            if operator is None:
                if key not in tags:
                    return False
            elif operator == "!~":
                if key in tags and re.search(value, tags[key]):
                    return False
            elif operator == "~":
                if key not in tags or not re.search(value, tags[key]):
                    return False
            elif operator == "=":
                if tags.get(key) != value:
                    return False
            elif operator == "!=":
                if tags.get(key) == value:
                    return False
        return True
    return accepts


def extract_mode_xml(pbf_path: Path, destination: Path, mode: str, polygon):
    metadata_path = destination.with_suffix(".osm.json")
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if sha256_file(destination) != metadata["sha256"]:
            raise ValueError("Extract cache checksum mismatch")
        return metadata
    expression = ox._overpass._get_network_filter(mode)  # pinned OSMnx 2.1.0; covered by tests
    accepts = compile_osmnx_filter(expression)
    region = prep(polygon)
    west, south, east, north = polygon.bounds
    temporary = destination.with_suffix(".tmp.osm")
    count = 0
    with osmium.BackReferenceWriter(str(temporary), ref_src=str(pbf_path), overwrite=True, remove_tags=False) as writer:
        class Roads(osmium.SimpleHandler):
            def way(self, way):
                nonlocal count
                if "highway" not in way.tags:
                    return
                tags = dict(way.tags)
                if not accepts(tags):
                    return
                coords = [(n.lon, n.lat) for n in way.nodes if n.location.valid()]
                if len(coords) < 2 or len(coords) != len(way.nodes):
                    return
                xs, ys = zip(*coords)
                if max(xs) < west or min(xs) > east or max(ys) < south or min(ys) > north:
                    return
                if region.intersects(LineString(coords)):
                    writer.add_way(way)
                    count += 1
        Roads().apply_file(str(pbf_path), locations=True)
    temporary.replace(destination)
    metadata = {"ways": count, "filter": expression, "sha256": sha256_file(destination),
                "method": "Pyosmium geometry-intersecting whole ways plus referenced nodes; OSMnx mode filter"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    from .paths import RAW_DIR
    print(download_snapshot(RAW_DIR / "travel_network/new-zealand-latest.osm.pbf"))
