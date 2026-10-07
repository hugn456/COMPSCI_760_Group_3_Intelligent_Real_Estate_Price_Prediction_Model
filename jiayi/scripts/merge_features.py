"""Offline, provenance-checked merge of the two Jiayi feature columns."""
import argparse
from pathlib import Path
import pandas as pd
from prepare_jiayi_data import digest

FEATURES = ['noise_category_combination', 'sa2_new_dwellings_consented_12m']
QUALITY = ['noise_match_status', 'noise_temporal_status', 'consents_match_status',
           'consents_release_id', 'consents_geography_version', 'consents_published_at',
           'consents_sa2_code', 'consents_boundary_distance_m']


def merge(source_path, features_path, output_path):
    source_path, features_path, output_path = map(Path, [source_path, features_path, output_path])
    if output_path.resolve() in {source_path.resolve(), features_path.resolve()}:
        raise ValueError('Choose an output distinct from both inputs.')
    source, features = pd.read_parquet(source_path), pd.read_parquet(features_path)
    if not source.index.is_unique or features.source_index.duplicated().any():
        raise ValueError('Duplicate merge keys.')
    if features.source_sha256.isna().any() or set(features.source_sha256) != {digest(source_path)}:
        raise ValueError('Features were generated from a different source snapshot; rebuild after coordinate updates.')
    if set(source.index) != set(features.source_index):
        raise ValueError('Source index coverage differs.')
    columns = FEATURES + [c for c in QUALITY if c in features]
    overlap = set(columns).intersection(source.columns)
    if overlap:
        raise ValueError(f'Input already contains derived columns: {sorted(overlap)}. Use the clean source.')
    merged = source.join(features.set_index('source_index')[columns], how='left', validate='one_to_one')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=True)
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.source, args.features, args.output)
    print(f'Merged {len(result):,} rows; original source index retained.')
