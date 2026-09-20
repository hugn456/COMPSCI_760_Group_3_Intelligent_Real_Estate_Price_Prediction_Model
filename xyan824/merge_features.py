"""Standalone, offline, hash-checked feature merge. Requires pandas and pyarrow."""
import argparse
import hashlib
from pathlib import Path
import pandas as pd


def merge(source_path, features_path, output_path):
    source_path, features_path, output_path = map(Path, [source_path, features_path, output_path])
    if output_path.resolve() in {source_path.resolve(), features_path.resolve()}:
        raise ValueError("Choose a separate output path; inputs must remain unchanged")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source, features = pd.read_parquet(source_path), pd.read_parquet(features_path)
    if features.source_sha256.isna().any() or set(features.source_sha256) != {digest}:
        raise ValueError("This feature table belongs to a different source Parquet snapshot")
    if not source.index.is_unique or features.source_index.duplicated().any():
        raise ValueError("Merge keys are not unique")
    if set(features.source_index) != set(source.index):
        raise ValueError("Expected full source-index coverage; refuse partial or mismatched merge")
    columns = [c for c in features if c.endswith(("_network_m", "_euclidean_m", "_status", "_distance_band", "_snap_m", "_boundary_sensitive"))
               or "_within_" in c or c in {"source_index", "record_id", "source_sha256", "retail_snapshot_outside_extent"}]
    merged = source.join(features[columns].set_index("source_index"), how="left", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=True)
    print(f"Merged {len(merged):,} rows; original index retained: {output_path}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge(args.source, args.features, args.output)
