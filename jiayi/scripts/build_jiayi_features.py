"""Generate row-preserving noise and release-specific consent features locally."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from prepare_jiayi_data import ROOT, OUT, digest, write_json
from check_jiayi_spatial_sample import classify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'data/private/dvrs_processed.parquet')
    args=parser.parse_args()
    source=args.input.resolve()
    frame=pd.read_parquet(OUT/'property_input.parquet')
    assert frame.source_sha256.eq(digest(source)).all() and frame.record_id.is_unique
    frame=frame.set_index('source_index',drop=False)
    valid=frame.coordinate_valid
    points=gpd.GeoDataFrame(frame.loc[valid].copy(),geometry=gpd.points_from_xy(frame.loc[valid,'longitude'],frame.loc[valid,'latitude']),crs=4326).to_crs(2193)
    noise=gpd.read_parquet(OUT/'aircraft_noise_operative.parquet')
    nh=classify(points,noise,'TYPE')
    labels=noise.set_index('TYPE').type_label.to_dict()
    frame['noise_category_combination']=pd.Series(pd.NA,index=frame.index,dtype='string')
    frame['noise_match_status']='invalid_coordinate'
    frame['noise_types_raw']='[]'
    frame.loc[valid,'noise_types_raw']=nh.hits.map(json.dumps)
    frame.loc[valid,'noise_category_combination']=nh.hits.map(lambda h:' | '.join(labels[x] for x in h) if h else 'outside_current_overlay')
    frame.loc[valid,'noise_match_status']=nh.hits.map(lambda h:'matched_current_snapshot' if h else 'outside_current_overlay')
    frame.loc[nh.index[nh.boundary],'noise_category_combination']=pd.NA
    frame.loc[nh.index[nh.boundary],'noise_match_status']='boundary_ambiguous'
    frame['noise_temporal_status']='long_term_planning_proxy_current_snapshot'
    frame['noise_snapshot_sha256']=digest(OUT/'aircraft_noise_operative.parquet')
    print('Noise linked',flush=True)
    releases=pd.read_parquet(OUT/'consents_linkage_release_index.parquet')
    releases['available']=pd.to_datetime(releases.earliest_usable_sale_date)
    linked=pd.merge_asof(frame[['source_index','sale_date']].reset_index(drop=True).sort_values('sale_date'),releases.sort_values('available'),left_on='sale_date',right_on='available',direction='backward').set_index('source_index').reindex(frame.index)
    frame['consents_release_id']=linked.release_id
    frame['consents_geography_version']=linked.sa2_version
    frame['consents_published_at']=linked.published_at_nz
    frame['consents_sa2_code']=pd.Series(pd.NA,index=frame.index,dtype='string')
    frame['consents_match_status']='no_prior_release'
    frame['consents_boundary_distance_m']=np.nan
    for version in ['2018','2023']:
        polygons=gpd.read_parquet(OUT/f'sa2_{version}_auckland.parquet')
        use=frame.index[frame.consents_geography_version.eq(version)&valid]
        subset=points.loc[use]
        hit=classify(subset,polygons,'sa2_code')
        frame.loc[use,'consents_match_status']='outside_or_multiple_sa2'
        good=hit.index[hit.hits.map(len).eq(1)&~hit.boundary]
        frame.loc[good,'consents_sa2_code']=hit.loc[good,'hits'].map(lambda h:h[0])
        frame.loc[good,'consents_match_status']='matched_release_specific_snapshot'
        frame.loc[hit.index[hit.boundary],'consents_match_status']='boundary_ambiguous'
        bycode=polygons.set_index('sa2_code')
        for code,idx in frame.loc[good].groupby('consents_sa2_code').groups.items():
            area=bycode.loc[code]
            frame.loc[idx,'consents_boundary_distance_m']=points.loc[idx].distance(area.geometry.boundary)
            if area.geometry_repaired:
                frame.loc[idx,'consents_match_status']='geometry_repair_requires_review'
        print('SA2',version,'linked',len(use),flush=True)
    frame.loc[~valid,'consents_match_status']='invalid_coordinate'
    totals=pd.read_parquet(OUT/'consents_12m_by_release_sa2.parquet')
    lookup=totals.set_index(['release_id','sa2_version','sa2_code']).new_dwellings_consented_12m
    assert lookup.index.is_unique
    keys=pd.MultiIndex.from_frame(frame[['consents_release_id','consents_geography_version','consents_sa2_code']])
    frame['sa2_new_dwellings_consented_12m']=pd.array(lookup.reindex(keys).to_numpy(),dtype='Int64')
    eligible=frame.consents_match_status.eq('matched_release_specific_snapshot')
    frame.loc[eligible&frame.sa2_new_dwellings_consented_12m.isna(),'consents_match_status']='incomplete_window'
    frame.loc[~eligible,'sa2_new_dwellings_consented_12m']=pd.NA
    used=frame.consents_published_at.notna()
    assert (pd.to_datetime(frame.loc[used,'consents_published_at']).dt.normalize()<frame.loc[used,'sale_date']).all()
    assert len(frame)==len(pd.read_parquet(source)) and frame.record_id.is_unique
    # Compare the independent full-data path against the previously reviewed sample.
    sample=pd.read_parquet(OUT/'spatial_sample_200.parquet').set_index('source_index')
    for col in ['noise_category_combination','sa2_new_dwellings_consented_12m']:
        pd.testing.assert_series_equal(frame.loc[sample.index,col].astype('string'),sample[col].astype('string'),check_names=False)
    dest=ROOT/'data/processed/jiayi';dest.mkdir(parents=True,exist_ok=True)
    frame.reset_index(drop=True).to_parquet(dest/'dvrs_noise_consents_features.parquet',index=False)
    report={'rows':len(frame),'noise_status':frame.noise_match_status.value_counts().to_dict(),'noise_categories':frame.noise_category_combination.value_counts().to_dict(),'consents_status':frame.consents_match_status.value_counts().to_dict(),'sample_200_agrees':True,'source_sha256':digest(source)}
    write_json(ROOT/'reports/jiayi_full_feature_quality.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
