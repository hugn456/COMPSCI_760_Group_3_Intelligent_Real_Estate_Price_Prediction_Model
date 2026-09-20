"""Reproducible local walk/drive distances; no Council rows leave this process."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Point, box

from .boundary import load_boundary
from .council_adapter import load_council
from .features import _nearest_distances, distance_bands, presence_flag
from .manifest import sha256_file, utc_now
from .paths import ROOT, load_json_config
from .quality import write_quality_report
from .travel_network import ensure_graph, study_area
from .travel_compact import CompactGraph


def reversed_min_matrix(graph: nx.MultiDiGraph):
    """Collapse parallel edges by MIN (sparse conversion would otherwise SUM)."""
    if isinstance(graph, CompactGraph):
        nodes = graph.node_ids
        return nodes, {int(n): i for i, n in enumerate(nodes)}, graph.reverse
    nodes = np.array(sorted(graph.nodes), dtype=np.int64)
    lookup = {int(n): i for i, n in enumerate(nodes)}
    rows, cols, weights = [], [], []
    for u, neighbors in graph.adjacency():
        for v, edges in neighbors.items():
            values = [float(data["length"]) for data in edges.values()]
            if any(not np.isfinite(w) or w < 0 for w in values):
                raise ValueError("Every edge must have a finite nonnegative length")
            rows.append(lookup[v])  # Reverse direction: destination -> origin.
            cols.append(lookup[u])
            weights.append(min(values))
    matrix = csr_matrix((weights, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=float)
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    return nodes, lookup, matrix


def nearest_facilities(matrix, source_indices):
    """SciPy's existing multi-source Dijkstra returns distance AND winning source.

    min_only avoids an all-pairs matrix. Predecessors permit sampled route QA.
    Equal-distance source ties may vary across SciPy versions; distances don't.
    """
    indices = np.unique(np.asarray(source_indices, dtype=np.int32))
    if len(indices) == 0:
        size = matrix.shape[0]
        return np.full(size, np.inf), np.full(size, -9999), np.full(size, -9999)
    return dijkstra(matrix, directed=True, indices=indices, min_only=True, return_predecessors=True)


def route_from_predecessors(origin, owner, predecessors):
    route = [int(origin)]
    while route[-1] != owner:
        nxt = int(predecessors[route[-1]])
        if nxt < 0 or len(route) > len(predecessors):
            raise ValueError("Invalid predecessor chain")
        route.append(nxt)
    return route


def points(frame):
    return gpd.GeoDataFrame(frame.copy(), geometry=gpd.points_from_xy(frame.longitude, frame.latitude), crs=4326)


def snap_nodes(graph, longitude, latitude):
    if isinstance(graph, CompactGraph):
        return graph.nearest_nodes(longitude, latitude)
    return ox.distance.nearest_nodes(graph, X=longitude, Y=latitude, return_dist=True)


def summarize_distances(series):
    valid = series.dropna()
    return {"available": int(len(valid)), "missing": int(series.isna().sum()),
            "zero": int((valid == 0).sum()), "unique": int(valid.nunique()),
            "quantiles_m": {str(k): float(v) for k, v in valid.quantile([0, .25, .5, .75, .95, .99, 1]).items()}}


def save_route_maps(graph, examples, output, mode):
    """Public roads only; no property address, price, or original coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    selected = examples[:12]
    if not selected:
        return
    if isinstance(graph, CompactGraph):
        xy = np.column_stack([graph.longitude, graph.latitude])
        edge = graph.forward.tocoo()
        segments = np.stack([xy[edge.row], xy[edge.col]], axis=1)
    else:
        all_nodes = list(graph.nodes)
        node_lookup = {n: i for i, n in enumerate(all_nodes)}
        xy = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in all_nodes])
        segments = np.array([[xy[node_lookup[u]], xy[node_lookup[v]]] for u, v in graph.edges()])
    for page in range(0, len(selected), 6):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, example in zip(axes.flat, selected[page:page + 6]):
            route = example["route_nodes"]
            coords = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in route])
            low, high = coords.min(axis=0) - .002, coords.max(axis=0) + .002
            keep = ((segments[:, :, 0].max(axis=1) >= low[0]) & (segments[:, :, 0].min(axis=1) <= high[0])
                    & (segments[:, :, 1].max(axis=1) >= low[1]) & (segments[:, :, 1].min(axis=1) <= high[1]))
            ax.add_collection(LineCollection(segments[keep], colors="#cccccc", linewidths=.5))
            ax.plot(coords[:, 0], coords[:, 1], color="#1565c0", linewidth=2)
            ax.scatter(*coords[0], color="#1565c0", s=25)
            ax.scatter(*coords[-1], color="#d35400", s=25)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                   title=f'{mode} / {example["category"]} / {example["distance_m"]:.0f} m')
            ax.set_aspect(1 / np.cos(np.deg2rad(coords[:, 1].mean())))
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(selected[page:page + 6]):]:
            ax.axis("off")
        fig.suptitle("Snapped-node route QA — blue: origin node; orange: facility node\n"
                     "OSM contributors / ODbL. Node-to-node approximation, not door-to-door navigation.")
        fig.tight_layout()
        fig.savefig(output / f"{mode}_routes_{page // 6 + 1}.png", dpi=140)
        plt.close(fig)


def build_travel_features(config_path: Path, scope="pilot", buffer_m=None, sample_size=None) -> Path:
    started = perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    effective_buffer = config["buffer_m"] if buffer_m is None else buffer_m
    name = f"{scope}_{int(effective_buffer)}m"
    output = ROOT / "data" / "processed" / "travel_distance" / name
    report_dir = ROOT / "reports" / f"travel_distance_{name}"
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame, input_meta = load_council((ROOT / config["input"]).resolve())
    facilities_path = (ROOT / config["facilities"]).resolve()
    facilities = pd.read_parquet(facilities_path)
    facilities = facilities[facilities.amenity_type.isin(config["categories"])].copy()
    facilities["facility_key"] = facilities.source.astype(str) + ":" + facilities.source_id.astype(str)
    facilities = facilities.sort_values("facility_key").drop_duplicates(["facility_key", "amenity_type"])
    for col in ["latitude", "longitude"]:
        facilities[col] = pd.to_numeric(facilities[col], errors="coerce")
    facilities = facilities.dropna(subset=["latitude", "longitude"])
    core, polygon = study_area(config, scope, effective_buffer)
    valid = frame.coordinate_valid
    projected = points(frame.loc[valid]).to_crs(2193)
    boundary = load_boundary().to_crs(2193).geometry.union_all()
    frame["inside_auckland"] = False
    frame.loc[valid, "inside_auckland"] = projected.geometry.covered_by(boundary).to_numpy()
    in_core = pd.Series(False, index=frame.index)
    in_core.loc[valid] = projected.geometry.covered_by(core).to_numpy()
    if scope == "pilot":
        eligible = frame.index[in_core & frame.inside_auckland]
        count = config["pilot_size"] if sample_size is None else sample_size
        selected = pd.Series(eligible).sample(n=min(count, len(eligible)), random_state=config["seed"]).sort_values()
        frame = frame.loc[selected.to_numpy()].copy()
        if frame.empty:
            raise ValueError("No pilot records in study area")
    frame = frame.reset_index(drop=True)
    eligible = frame.coordinate_valid & frame.inside_auckland
    projected = points(frame.loc[eligible]).to_crs(2193)
    retail_extent_distance = None
    if "retail_snapshot_bounds_wsen" in config:
        retail_region = gpd.GeoSeries([box(*config["retail_snapshot_bounds_wsen"])], crs=4326).to_crs(2193).iloc[0]
        frame["retail_snapshot_outside_extent"] = True
        frame.loc[eligible, "retail_snapshot_outside_extent"] = ~projected.geometry.covered_by(retail_region).to_numpy()
        retail_extent_distance = projected.distance(retail_region.boundary).reindex(frame.index)
    facility_points = points(facilities).to_crs(2193)
    polygon_m = gpd.GeoSeries([polygon], crs=4326).to_crs(2193).iloc[0]
    # Straight-line baselines use exactly the same geographically eligible facility pool.
    facility_points = facility_points[facility_points.geometry.covered_by(polygon_m)].copy()
    if facility_points.empty:
        raise ValueError("No facilities within the graph query area")
    band_config = load_json_config("amenities.json")
    for category in config["categories"]:
        distances = _nearest_distances(projected, facility_points[facility_points.amenity_type == category])
        frame[f"{category}_euclidean_m"] = distances.reindex(frame.index)
        frame[f"{category}_distance_band"] = distance_bands(frame[f"{category}_euclidean_m"], band_config["feature_bands_km"][category])
        threshold = band_config["presence_thresholds_km"][category]
        label = f"{int(threshold * 1000)}m" if threshold < 1 else f"{threshold:g}km"
        frame[f"{category}_within_{label}"] = presence_flag(frame[f"{category}_euclidean_m"], threshold)
    metadata = {"created_at": utc_now(), "input": input_meta, "config": config, "scope": scope,
                "effective_buffer_m": effective_buffer, "selected_rows": len(frame),
                "facility_sha256": sha256_file(facilities_path), "graphs": {},
                "versions": {p: importlib.metadata.version(p) for p in ["osmnx", "networkx", "scipy", "pandas", "geopandas", "numpy"]},
                "distance_definition": "Shortest snapped-node road distance; access connectors excluded",
                "facility_scope": "Existing AT and selected OSM chain snapshot, restricted to query polygon; not all amenities",
                "tie_policy": "Same-node facilities: sorted facility_key. Equal-distance different nodes: SciPy tie, version recorded",
                "snap_thresholds": "Provisional QA cutoffs; missing if exceeded, not a geocoding accuracy guarantee"}
    audit_tables, route_checks = [], []
    for mode in config["modes"]:
        print(f"START {mode}: {len(frame)} property rows", flush=True)
        graph_path, graph_meta = ensure_graph(config_path, mode, scope, effective_buffer)
        graph = CompactGraph.load(graph_path) if graph_path.suffix == ".npz" else ox.io.load_graphml(graph_path)
        metadata["graphs"][mode] = {**graph_meta, "path": str(graph_path)}
        nodes, lookup, matrix = reversed_min_matrix(graph)
        prop_nodes, prop_snap = snap_nodes(graph, frame.loc[eligible, "longitude"], frame.loc[eligible, "latitude"])
        snap = pd.Series(prop_snap, index=frame.index[eligible])
        prop_node = pd.Series(np.asarray(prop_nodes, dtype=np.int64), index=frame.index[eligible])
        frame[f"{mode}_origin_snap_m"] = snap.reindex(frame.index)
        f_nodes, f_snap = snap_nodes(graph, facility_points.to_crs(4326).geometry.x, facility_points.to_crs(4326).geometry.y)
        snapped_facilities = facility_points.assign(node_id=np.asarray(f_nodes, dtype=np.int64), snap_m=f_snap)
        max_snap = config["max_snap_m"][mode]
        graph_examples = []
        for category in config["categories"]:
            prefix = f"{mode}_{category}"
            candidates = snapped_facilities[snapped_facilities.amenity_type == category]
            accepted = candidates[candidates.snap_m <= max_snap].sort_values("facility_key").drop_duplicates("node_id")
            source_indices = [lookup[n] for n in accepted.node_id]
            dist, pred, owner = nearest_facilities(matrix, source_indices)
            status = pd.Series("invalid_coordinate", index=frame.index, dtype="string")
            status.loc[frame.coordinate_valid & ~frame.inside_auckland] = "outside_auckland"
            status.loc[eligible] = "no_reachable_facility" if len(accepted) else "no_eligible_facility"
            status.loc[snap.index[snap > max_snap]] = "origin_snap_too_far"
            origin_indices = prop_node.map(lookup).astype(int)
            reachable_rows = origin_indices.index[(snap <= max_snap) & np.isfinite(dist[origin_indices.to_numpy()])]
            distance = pd.Series(np.nan, index=frame.index)
            distance.loc[reachable_rows] = dist[origin_indices.loc[reachable_rows].to_numpy()]
            status.loc[reachable_rows] = "ok"
            status.loc[distance.eq(0)] = "same_node_approximation"
            frame[f"{prefix}_network_m"] = distance
            frame[f"{prefix}_status"] = status
            if category == "supermarket" and retail_extent_distance is not None:
                frame[f"{prefix}_snapshot_boundary_sensitive"] = (
                    frame.retail_snapshot_outside_extent | distance.ge(retail_extent_distance)
                )
            # Does route length permit reaching the query boundary? Conservative flag.
            snapped_xy = gpd.GeoSeries([Point(graph.nodes[n]["x"], graph.nodes[n]["y"])
                                       for n in prop_node], crs=4326).to_crs(2193)
            boundary_dist = pd.Series(snapped_xy.distance(polygon_m.boundary).to_numpy(), index=prop_node.index)
            frame[f"{prefix}_boundary_sensitive"] = (distance >= boundary_dist.reindex(frame.index)).fillna(False)
            audit = frame[["record_id", "source_index"]].copy()
            audit["mode"], audit["category"] = mode, category
            audit["network_m"], audit["status"] = distance, status
            audit["origin_node"] = prop_node.reindex(frame.index).astype("Int64")
            audit["origin_snap_m"] = snap.reindex(frame.index)
            audit["boundary_sensitive"] = frame[f"{prefix}_boundary_sensitive"]
            accepted_by_index = {lookup[int(row.node_id)]: row for row in accepted.itertuples()}
            for col in ["facility_key", "facility_geometry_method"]:
                audit[col] = pd.Series(pd.NA, index=frame.index, dtype="string")
            audit["facility_node"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
            audit["facility_snap_m"] = np.nan
            winner_indices = pd.Series(owner[origin_indices.loc[reachable_rows].to_numpy()], index=reachable_rows)
            for target, attribute in [("facility_key", "facility_key"), ("facility_geometry_method", "geometry_method"),
                                      ("facility_node", "node_id"), ("facility_snap_m", "snap_m")]:
                mapping = {i: getattr(row, attribute) for i, row in accepted_by_index.items()}
                audit.loc[reachable_rows, target] = winner_indices.map(mapping).to_numpy()
            audit_tables.append(audit)
            # Fixed seeded ordinary routes + longest/zero cases, all checked against NetworkX.
            ordinary = reachable_rows[distance.loc[reachable_rows] > 0]
            selected = pd.Series(ordinary).sample(n=min(10, len(ordinary)), random_state=config["seed"]).tolist()
            selected += distance.dropna().nlargest(2).index.tolist()
            selected += distance[distance.eq(0)].index[:1].tolist()
            for row_id in dict.fromkeys(selected):
                oi = int(origin_indices.loc[row_id]); wi = int(owner[oi])
                route_idx = route_from_predecessors(oi, wi, pred)
                route_nodes = [int(nodes[i]) for i in route_idx]
                verified = (graph.verify_route(route_nodes[0], route_nodes[-1], distance.loc[row_id])
                            if isinstance(graph, CompactGraph) else
                            nx.shortest_path_length(graph, route_nodes[0], route_nodes[-1], weight="length"))
                error = abs(verified - distance.loc[row_id])
                if error > 1e-5:
                    raise AssertionError(f"Shortest path disagreement: {error}")
                check = {"mode": mode, "category": category, "source_index": int(frame.loc[row_id, "source_index"]),
                         "distance_m": float(distance.loc[row_id]), "networkx_error_m": error,
                         "route_nodes": route_nodes}
                route_checks.append(check)
                if len(graph_examples) < 6 or (category == config["categories"][-1] and len(graph_examples) < 12):
                    graph_examples.append(check)
            metadata.setdefault("facility_matching", {})[prefix] = {
                "candidates": len(candidates), "snap_accepted": int((candidates.snap_m <= max_snap).sum()),
                "distinct_destination_nodes": len(accepted), "snap_quantiles_m": summarize_distances(candidates.snap_m)}
            print(f"RESULT {prefix}: {status.value_counts().to_dict()}", flush=True)
        save_route_maps(graph, graph_examples, report_dir, mode)
        del graph, matrix
        gc.collect()
    features_path = output / "dvrs_travel_features.parquet"
    frame.to_parquet(features_path, index=False)
    frame.to_csv(output / "dvrs_travel_features.csv", index=False)
    pd.concat(audit_tables, ignore_index=True).to_parquet(report_dir / "travel_distance_audit.parquet", index=False)
    # Route traces include source_index and are team-only audit material, not publication figures.
    (report_dir / "route_checks.json").write_text(json.dumps(route_checks, indent=2), encoding="utf-8")
    metadata["elapsed_seconds"] = round(perf_counter() - started, 2)
    metadata["features_sha256"] = sha256_file(features_path)
    metadata["route_check_count"] = len(route_checks)
    metadata["inside_auckland_rows"] = int(frame.inside_auckland.sum())
    if "retail_snapshot_outside_extent" in frame:
        metadata["retail_snapshot_outside_extent_rows"] = int(frame.retail_snapshot_outside_extent.sum())
    metadata["distance_distributions"] = {c: summarize_distances(frame[c]) for c in frame if c.endswith(("_network_m", "_euclidean_m"))}
    metadata["statuses"] = {c: frame[c].value_counts().to_dict() for c in frame if c.endswith("_status")}
    metadata["boundary_sensitive_counts"] = {c: int(frame[c].sum()) for c in frame if c.endswith("_boundary_sensitive")}
    metadata["limitations"] = [
        "Node-to-node approximation; no verified door access or complete turn restrictions.",
        "Coordinates are geocoder output; high confidence is not proof of correct address.",
        "Supermarket coverage is selected chains from the existing snapshot, not every supermarket.",
        "Drive-to-bus-stop is exploratory road proximity, not parking or transit journey time.",
        "Same-node zeros and provisional snap cutoffs require sensitivity review.",
        "Route samples numerically checked; map review is not independent ground truth.",
        "No house-price models fitted; team split and model code still required.",
    ]
    write_quality_report("travel_distance_quality", metadata, report_dir)
    (report_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"DONE {features_path}", flush=True)
    return features_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/travel_distance.json")
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--buffer-m", type=float)
    parser.add_argument("--sample-size", type=int)
    args = parser.parse_args()
    build_travel_features(args.config.resolve(), args.scope, args.buffer_m, args.sample_size)


if __name__ == "__main__":
    main()
