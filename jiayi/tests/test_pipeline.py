"""Synthetic local fixtures only; no team rows or online requests."""
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_jiayi_features as builder
from property_input import load_council
from prepare_jiayi_data import digest
from merge_features import merge


def test_full_linkage_and_updated_coordinate_hash(tmp_path, monkeypatch):
    source = tmp_path / 'source.parquet'
    pd.DataFrame({'longitude': [175., 175., np.nan, 176.],
                  'latitude': [-37., -37., -37., -37.],
                  'sale_date': ['03012024', '01012024', '03012024', '03012024']},
                 index=[10, 20, 30, 40]).to_parquet(source)
    frame, _ = load_council(source)
    out = tmp_path / 'interim'
    out.mkdir()
    frame.to_parquet(out / 'property_input.parquet', index=False)
    point = gpd.GeoSeries(gpd.points_from_xy([175], [-37]), crs=4326).to_crs(2193).iloc[0]
    polygon = box(point.x-100, point.y-100, point.x+100, point.y+100)
    gpd.GeoDataFrame({'TYPE': [1], 'type_label': ['synthetic_noise']},
                     geometry=[polygon], crs=2193).to_parquet(out / 'aircraft_noise_operative.parquet')
    for version in ['2018', '2023']:
        gpd.GeoDataFrame({'sa2_code': ['toy'], 'geometry_repaired': [False]},
                         geometry=[polygon], crs=2193).to_parquet(out / f'sa2_{version}_auckland.parquet')
    pd.DataFrame({'release_id': ['2023-12'], 'sa2_version': ['2023'],
                  'published_at_nz': ['2024-01-01'], 'earliest_usable_sale_date': ['2024-01-02']}).to_parquet(
                      out / 'consents_linkage_release_index.parquet')
    pd.DataFrame({'release_id': ['2023-12'], 'sa2_version': ['2023'],
                  'sa2_code': ['toy'], 'new_dwellings_consented_12m': [12]}).to_parquet(
                      out / 'consents_12m_by_release_sa2.parquet')
    pd.DataFrame({'source_index': [10, 20], 'noise_category_combination': ['synthetic_noise']*2,
                  'sa2_new_dwellings_consented_12m': pd.array([12, pd.NA], dtype='Int64')}).to_parquet(
                      out / 'spatial_sample_200.parquet')
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    monkeypatch.setattr(builder, 'OUT', out)
    monkeypatch.setattr(sys, 'argv', ['build_jiayi_features.py', '--input', str(source)])
    builder.main()
    features_path = tmp_path / 'data/processed/jiayi/dvrs_noise_consents_features.parquet'
    features = pd.read_parquet(features_path).set_index('source_index', drop=False)
    assert len(features) == 4 and features.index.tolist() == [10, 20, 30, 40]
    assert features.loc[10, 'sa2_new_dwellings_consented_12m'] == 12
    assert features.loc[20, 'consents_match_status'] == 'no_prior_release'
    assert pd.isna(features.loc[20, 'sa2_new_dwellings_consented_12m'])
    assert features.loc[30, 'noise_match_status'] == 'invalid_coordinate'
    assert features.loc[30, 'consents_match_status'] == 'invalid_coordinate'
    assert features.loc[40, 'consents_match_status'] == 'outside_or_multiple_sa2'
    merged = merge(source, features_path, tmp_path / 'merged.parquet')
    assert merged.index.tolist() == [10, 20, 30, 40]
    # Reusing old feature tables after a coordinate update must be rejected.
    changed = pd.read_parquet(source)
    changed.loc[10, 'longitude'] = 176
    updated_source = tmp_path / 'updated.parquet'
    changed.to_parquet(updated_source)
    with pytest.raises(ValueError, match='different source snapshot'):
        merge(updated_source, features_path, tmp_path / 'bad_merge.parquet')
    with pytest.raises(ValueError, match='distinct'):
        merge(source, features_path, source)


def test_input_requires_unique_original_index(tmp_path):
    source = tmp_path / 'duplicates.parquet'
    pd.DataFrame({'longitude': [1, 1], 'latitude': [1, 1], 'sale_date': ['01012024']*2},
                 index=[7, 7]).to_parquet(source)
    with pytest.raises(ValueError, match='not unique'):
        load_council(source)
