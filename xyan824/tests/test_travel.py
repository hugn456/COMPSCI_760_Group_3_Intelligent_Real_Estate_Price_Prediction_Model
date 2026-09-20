from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from auckland_accessibility.council_adapter import load_council, merge_travel_features
from auckland_accessibility.travel_features import reversed_min_matrix, nearest_facilities, route_from_predecessors


def test_directed_multisource_parallel_edges_and_disconnected_nodes():
    g = nx.MultiDiGraph()
    g.add_nodes_from([1, 2, 3, 4, 5, 6])
    # Destination 3 is directly reachable but expensive; 4 via 2 is nearer.
    g.add_edge(1, 3, length=100)
    g.add_edge(1, 2, length=3)
    g.add_edge(1, 2, length=50)  # parallel edges must not be summed
    g.add_edge(2, 4, length=4)
    g.add_edge(3, 5, length=1)  # 5 cannot travel backwards to 3
    nodes, lookup, matrix = reversed_min_matrix(g)
    distances, predecessors, owners = nearest_facilities(matrix, [lookup[3], lookup[4]])
    assert distances[lookup[1]] == 7
    assert nodes[owners[lookup[1]]] == 4
    assert np.isinf(distances[lookup[5]])
    assert np.isinf(distances[lookup[6]])
    assert distances[lookup[3]] == 0
    route = route_from_predecessors(lookup[1], lookup[4], predecessors)
    assert nodes[route].tolist() == [1, 2, 4]
    for n in g:
        expected = min((nx.shortest_path_length(g, n, target, weight="length")
                        if nx.has_path(g, n, target) else np.inf) for target in [3, 4])
        assert distances[lookup[n]] == expected


def test_equal_distance_and_no_facilities():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, length=5)
    graph.add_edge(1, 3, length=5)
    nodes, lookup, matrix = reversed_min_matrix(graph)
    d, _, owner = nearest_facilities(matrix, [lookup[2], lookup[3]])
    assert d[lookup[1]] == 5
    assert nodes[owner[lookup[1]]] in [2, 3]
    d, _, owner = nearest_facilities(matrix, [])
    assert np.isinf(d).all()
    assert (owner == -9999).all()


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_bad_weights_rejected(bad):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, length=bad)
    with pytest.raises(ValueError):
        reversed_min_matrix(graph)


def make_source(path: Path):
    original = pd.DataFrame({"longitude": ["174.7", "174.7", "bad"], "latitude": ["-36.8"] * 3,
                             "geocode_confidence": [.7, .7, .4], "sale_date": ["1042025", "1042025", "bad"]},
                            index=[19, 37, 52])
    original.to_parquet(path)
    return original


def test_adapter_preserves_duplicate_sales_and_exact_index(tmp_path):
    path = tmp_path / "source.parquet"
    original = make_source(path)
    features, metadata = load_council(path)
    assert len(features) == 3
    assert metadata["exact_duplicate_rows"] == 1
    assert features.source_index.tolist() == [19, 37, 52]
    assert features.record_id.is_unique
    assert features.sale_date.iloc[0] == pd.Timestamp("2025-04-01")
    assert not features.coordinate_valid.iloc[2]
    features["walk_supermarket_network_m"] = [1., 2., np.nan]
    merged = merge_travel_features(original, features.iloc[[1, 0, 2]], path)
    assert merged.walk_supermarket_network_m.iloc[:2].tolist() == [1., 2.]
    assert merged.index.equals(original.index)
    with pytest.raises(ValueError, match="original source rows"):
        merge_travel_features(original.reset_index(drop=True), features, path)
    features.loc[0, "source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA256"):
        merge_travel_features(original, features, path)


def test_duplicate_index_rejected(tmp_path):
    path = tmp_path / "source.parquet"
    original = make_source(path)
    original.index = [0, 0, 1]
    original.to_parquet(path)
    with pytest.raises(ValueError, match="not unique"):
        load_council(path)


def test_graph_cache_separates_modes_and_config():
    from shapely.geometry import box
    from auckland_accessibility.travel_network import graph_paths
    config = {"simplify": False, "overpass_url": "https://example.invalid/api"}
    polygon = box(174.7, -36.9, 174.8, -36.8)
    a, _, _ = graph_paths(config, "walk", polygon)
    b, _, _ = graph_paths(config, "drive", polygon)
    c, _, _ = graph_paths({**config, "simplify": True}, "walk", polygon)
    d, _, _ = graph_paths(config, "walk", polygon.buffer(.01))
    assert len({a, b, c, d}) == 4


def test_pbf_filters_match_pinned_mode_semantics():
    import osmnx as ox
    from auckland_accessibility.travel_pbf import compile_osmnx_filter
    walk = compile_osmnx_filter(ox._overpass._get_network_filter("walk"))
    drive = compile_osmnx_filter(ox._overpass._get_network_filter("drive"))
    assert walk({"highway": "footway"})
    assert not drive({"highway": "footway"})
    assert drive({"highway": "motorway"})
    assert not walk({"highway": "motorway"})
    assert not walk({"highway": "residential", "access": "private"})
    assert not drive({"highway": "residential", "motor_vehicle": "no"})
    assert not walk({"highway": "residential", "foot": "no"})
    assert not walk({"amenity": "parking"})
    with pytest.raises(ValueError):
        compile_osmnx_filter('["highway"];unexpected')


def test_offline_end_to_end_export_and_validated_merge(tmp_path, monkeypatch):
    import json
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import box
    import auckland_accessibility.travel_features as tf

    source_path = tmp_path / "input.parquet"
    source = make_source(source_path)
    amenities = pd.DataFrame({"longitude": [174.7001, 174.7002], "latitude": [-36.8, -36.8],
                             "source": ["osm", "at_gtfs"], "source_id": ["node/2", "stop/3"],
                             "amenity_type": ["supermarket", "bus_stop"], "geometry_method": ["node", "stop"]})
    amenities.to_parquet(tmp_path / "facilities.parquet")
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for i, lon in enumerate([174.7, 174.7001, 174.7002], 1):
        graph.add_node(i, x=lon, y=-36.8)
    graph.add_edge(1, 2, length=10.)
    graph.add_edge(2, 3, length=10.)
    graph_path = tmp_path / "roads.graphml"
    ox.io.save_graphml(graph, graph_path)
    config = {"input": str(source_path), "facilities": str(tmp_path / "facilities.parquet"),
              "categories": ["supermarket", "bus_stop"], "modes": ["walk", "drive"],
              "max_snap_m": {"walk": 150, "drive": 200}, "seed": 760, "buffer_m": 5000}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    region = gpd.GeoDataFrame(geometry=[box(174.6, -36.9, 174.8, -36.7)], crs=4326)
    monkeypatch.setattr(tf, "ROOT", tmp_path)
    monkeypatch.setattr(tf, "load_boundary", lambda: region)
    monkeypatch.setattr(tf, "study_area", lambda *args: (region.to_crs(2193).geometry.iloc[0], region.geometry.iloc[0]))
    monkeypatch.setattr(tf, "ensure_graph", lambda *args: (graph_path, {}))
    monkeypatch.setattr(tf, "save_route_maps", lambda *args: None)
    result_path = tf.build_travel_features(cfg, scope="full")
    result = pd.read_parquet(result_path)
    assert result.source_index.tolist() == [19, 37, 52]
    assert result.walk_supermarket_network_m.iloc[:2].tolist() == [10., 10.]
    assert result.drive_bus_stop_network_m.iloc[:2].tolist() == [20., 20.]
    assert result.walk_supermarket_status.iloc[2] == "invalid_coordinate"
    assert pd.isna(result.walk_supermarket_network_m.iloc[2])
    assert len(merge_travel_features(source, result, source_path)) == 3
    audit = pd.read_parquet(tmp_path / "reports/travel_distance_full_5000m/travel_distance_audit.parquet")
    assert len(audit) == 12
    assert set(audit.loc[audit.status == "ok", "facility_key"]) == {"osm:node/2", "at_gtfs:stop/3"}


@pytest.mark.parametrize("mode", ["walk", "drive"])
def test_compact_topology_matches_osmnx_and_roundtrips(tmp_path, mode):
    import osmnx as ox
    from auckland_accessibility.travel_compact import compact_from_xml, CompactGraph
    path = tmp_path / "roads.osm"
    path.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="offline-test">
<node id="1" lat="-36.8" lon="174.7"/>
<node id="2" lat="-36.8" lon="174.7001"/>
<node id="3" lat="-36.8" lon="174.7002"/>
<node id="4" lat="-36.8" lon="174.7003"/>
<node id="5" lat="-36.8" lon="174.7004"/>
<way id="11"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/><tag k="oneway" v="yes"/></way>
<way id="12"><nd ref="2"/><nd ref="3"/><tag k="highway" v="residential"/><tag k="oneway" v="-1"/></way>
<way id="13"><nd ref="3"/><nd ref="4"/><tag k="highway" v="residential"/></way>
<way id="14"><nd ref="4"/><nd ref="5"/><tag k="highway" v="residential"/><tag k="junction" v="roundabout"/></way>
<way id="15"><nd ref="3"/><nd ref="4"/><tag k="highway" v="residential"/></way>
</osm>''', encoding="utf-8")
    baseline = ox.graph.graph_from_xml(path, bidirectional=(mode == "walk"), simplify=False, retain_all=True)
    compact = compact_from_xml(path, mode)
    compact.save(tmp_path / "roads.npz")
    compact = CompactGraph.load(tmp_path / "roads.npz")
    nodes, lookup, matrix = reversed_min_matrix(compact)
    for destination in nodes:
        distances, _, _ = nearest_facilities(matrix, [lookup[destination]])
        for origin in nodes:
            expected = (nx.shortest_path_length(baseline, origin, destination, weight="length")
                        if nx.has_path(baseline, origin, destination) else np.inf)
            assert distances[lookup[origin]] == pytest.approx(expected)
            if np.isfinite(expected):
                assert compact.verify_route(origin, destination, expected) == pytest.approx(expected)
    nearest, snap = compact.nearest_nodes([174.7001], [-36.8])
    assert nearest.tolist() == [2]
    assert snap[0] == pytest.approx(0)
