# Jay (xyan824): walking and driving distance calculation

Code only: no sales, coordinates, feature tables, road snapshots, model files or generated results are included.

Calculates shortest walking/driving road-network distances to supermarkets and bus stops. Separate OSMnx mode filters and road direction rules are used; Pyosmium reads the local OSM snapshot, and SciPy multi-source Dijkstra finds the nearest reachable facility. Existing distance helpers are reused. Output `source_index` matches the original sales DataFrame index, retained before any row filtering or reset.

## Local inputs (shared separately)

Place inputs under this folder; `data/` is ignored by Git:

- `data/private/dvrs_processed.parquet`: original indexed sales table, including longitude, latitude, geocode_confidence and sale_date (DDMMYYYY; leading-zero padding supported).
- `data/processed/amenities_points.parquet`: longitude, latitude, source, source_id, amenity_type (`supermarket` or `bus_stop`) and geometry_method. Use the agreed team facility snapshot.
- `data/processed/auckland_boundary.geojson`: Auckland study boundary, readable by GeoPandas.
- `data/raw/travel_network/new-zealand-latest.osm.pbf` and its JSON metadata: local public OSM snapshot. The optional download command below creates both; a current download will not reproduce an older snapshot exactly.

Paths and snap limits can be changed in `config/travel_distance.json`. Configuration coordinates describe the public study area, not individual properties.

## Run (PowerShell)

From the repository root:

```powershell
cd xyan824
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
# Optional: downloads a public NZ OSM snapshot. No property rows are uploaded.
& .venv/Scripts/python.exe -m auckland_accessibility.travel_pbf
& .venv/Scripts/python.exe -m auckland_accessibility.travel_features --scope pilot
& .venv/Scripts/python.exe -m auckland_accessibility.travel_features --scope full --buffer-m 10000
```

The boundary may instead be collected from the configured public source:

```powershell
& .venv/Scripts/python.exe -c "from auckland_accessibility.boundary import collect_boundary; from auckland_accessibility.http import HttpClient; collect_boundary(HttpClient(user_agent='COMPSCI760-travel-features'))"
```

Output is under `data/processed/travel_distance/`. Main columns are `walk_supermarket_network_m`, `walk_bus_stop_network_m`, `drive_supermarket_network_m`, and `drive_bus_stop_network_m`. Distances are metres between snapped road nodes, excluding access connectors. NaN means unavailable, not zero. Inspect accompanying status flags; `same_node_approximation` is not an exact door-to-door zero. Current snapshots represent retrospective accessibility, not historical travel conditions. Driving to a bus stop is road proximity, not a public-transport journey.

For an already generated feature file, merge before resetting the original sales index:

```powershell
python merge_features.py --source data/private/dvrs_processed.parquet --features path/to/dvrs_travel_features.parquet --output data/processed/dvrs_with_travel.parquet
```

The merge checks the source hash and one-to-one index coverage. Do not add generated data to this public repository.

## Offline checks

```powershell
$env:PYTHONPATH = 'src'
& .venv/Scripts/python.exe -m pytest tests/test_travel.py -q
```

Tests create synthetic fixtures locally; no team data or network access is required. They cover directionality, parallel edges, unreachable nodes, index-safe merging and comparison with OSMnx topology.

## References

- [OSMnx](https://github.com/gboeing/osmnx) and its [routing examples](https://github.com/gboeing/osmnx-examples)
- [SciPy multi-source Dijkstra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html)
- [Pyosmium](https://github.com/osmcode/pyosmium), [Geofabrik New Zealand](https://download.geofabrik.de/australia-oceania/new-zealand.html)

This folder contains the travel calculation and required helpers only, not the house-price modelling or Jiayi's feature pipeline.
