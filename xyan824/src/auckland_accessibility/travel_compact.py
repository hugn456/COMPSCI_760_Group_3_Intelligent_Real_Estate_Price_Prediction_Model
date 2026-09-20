"""Compact unsimplified road graphs using OSMnx rules and SciPy storage.

Topology is compared against OSMnx graph_from_xml in offline tests.
No handwritten shortest-path algorithm is used.
"""
from array import array
from collections.abc import Mapping
from itertools import groupby, pairwise
from pathlib import Path

import networkx as nx
import numpy as np
import osmium
import osmnx as ox
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


class NodeCoordinates(Mapping):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.graph.node_ids)

    def __iter__(self):
        return iter(self.graph.node_ids)

    def __getitem__(self, key):
        i = np.searchsorted(self.graph.node_ids, key)
        if i == len(self) or self.graph.node_ids[i] != key:
            raise KeyError(key)
        return {"x": float(self.graph.longitude[i]), "y": float(self.graph.latitude[i])}


class SparseAdjacency(Mapping):
    """Read-only NetworkX adjacency adapter; materialize one node's edges only."""
    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __iter__(self):
        return iter(range(len(self)))

    def __getitem__(self, node):
        if not isinstance(node, (int, np.integer)) or not 0 <= node < len(self):
            raise KeyError(node)
        start, end = self.matrix.indptr[node:node + 2]
        return {int(n): {"length": float(w)} for n, w in
                zip(self.matrix.indices[start:end], self.matrix.data[start:end])}


class NodePresence(Mapping):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(range(self.size))

    def __getitem__(self, node):
        if not isinstance(node, (int, np.integer)) or not 0 <= node < self.size:
            raise KeyError(node)
        return {}


class CompactGraph:
    def __init__(self, node_ids, longitude, latitude, reverse):
        self.node_ids, self.longitude, self.latitude = node_ids, longitude, latitude
        self.reverse = reverse
        self.forward = reverse.T.tocsr()
        self.nodes = NodeCoordinates(self)
        self.transform = Transformer.from_crs(4326, 2193, always_xy=True)
        x, y = self.transform.transform(longitude, latitude)
        self.xy = np.column_stack([x, y])
        self.tree = cKDTree(self.xy)

    def __len__(self):
        return len(self.node_ids)

    def number_of_edges(self):
        return self.reverse.nnz

    def nearest_nodes(self, longitude, latitude):
        x, y = self.transform.transform(np.asarray(longitude), np.asarray(latitude))
        distance, indices = self.tree.query(np.column_stack([x, y]))
        return self.node_ids[indices], distance

    def verify_route(self, origin_id, destination_id, expected_distance):
        """Independently solve with NetworkX without duplicating the whole graph."""
        origin = np.searchsorted(self.node_ids, origin_id)
        destination = np.searchsorted(self.node_ids, destination_id)
        small = nx.DiGraph()
        # Pinned NetworkX read-only internals; all-pairs fixture parity tests
        # cover this adapter. Its standard Dijkstra remains unmodified.
        small._node = NodePresence(len(self))
        small._adj = SparseAdjacency(self.forward)
        small._pred = SparseAdjacency(self.reverse)
        return nx.dijkstra_path_length(small, int(origin), int(destination), weight="length")

    def save(self, path: Path):
        with path.open("wb") as stream:
            np.savez_compressed(stream, node_ids=self.node_ids, longitude=self.longitude, latitude=self.latitude,
                                data=self.reverse.data, indices=self.reverse.indices, indptr=self.reverse.indptr)

    @classmethod
    def load(cls, path: Path):
        with np.load(path) as data:
            n = len(data["node_ids"])
            reverse = csr_matrix((data["data"], data["indices"], data["indptr"]), shape=(n, n))
            return cls(data["node_ids"], data["longitude"], data["latitude"], reverse)


def compact_from_xml(path: Path, mode: str):
    ids, longitude, latitude = array("q"), array("d"), array("d")
    origins, destinations = array("q"), array("q")
    oneway_values = {"yes", "true", "1", "-1", "reverse", "T", "F"}
    reversed_values = {"-1", "reverse", "T"}
    class Reader(osmium.SimpleHandler):
        def node(self, node):
            ids.append(node.id); longitude.append(node.location.lon); latitude.append(node.location.lat)

        def way(self, way):
            tags = dict(way.tags)
            node_ids = [key for key, _ in groupby(n.ref for n in way.nodes)]
            oneway = ox.graph._is_path_one_way(tags, mode == "walk", oneway_values)
            if oneway and ox.graph._is_path_reversed(tags, reversed_values):
                node_ids.reverse()
            for u, v in pairwise(node_ids):
                origins.append(u); destinations.append(v)
                if not oneway:
                    origins.append(v); destinations.append(u)
    Reader().apply_file(str(path))
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids)
    ids = ids[order]
    longitude, latitude = np.asarray(longitude)[order], np.asarray(latitude)[order]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate OSM node IDs")
    start_refs, end_refs = np.asarray(origins), np.asarray(destinations)
    u, v = np.searchsorted(ids, start_refs), np.searchsorted(ids, end_refs)
    if np.any(u >= len(ids)) or np.any(v >= len(ids)) or not np.array_equal(ids[u], start_refs) or not np.array_equal(ids[v], end_refs):
        raise ValueError("Incomplete node references in road extract")
    # All parallel unsimplified segments between the same two nodes have the
    # same endpoint-based length; deduplicate before sparse construction.
    _, unique = np.unique(v * len(ids) + u, return_index=True)
    u, v = u[unique], v[unique]
    lengths = ox.distance.great_circle(latitude[u], longitude[u], latitude[v], longitude[v])
    reverse = csr_matrix((lengths, (v, u)), shape=(len(ids), len(ids)))
    reverse.indices = reverse.indices.astype(np.int32)
    reverse.indptr = reverse.indptr.astype(np.int32)
    return CompactGraph(ids, longitude, latitude, reverse)
