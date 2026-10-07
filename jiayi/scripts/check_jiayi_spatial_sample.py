"""Source-backed SA2 collection and reproducible sample linkage, no price/model use."""
import argparse
import json
import pandas as pd
import geopandas as gpd
from prepare_jiayi_data import Sources, ROOT, OUT, RAW, write_json, normalize_release
from shapely import make_valid

SA2 = 'https://services2.arcgis.com/vKb0s8tBIA3bdocZ/ArcGIS/rest/services/Statistical_Area_2_2023/FeatureServer/0'


def collect(sources, version='2023'):
    url=SA2.replace('2023',version)
    field=f'SA2{version}_V1_00'
    name=field+'_NAME' if version=='2023' else 'SA22018_V1_NAME'
    sources.json(f'sa2/{version}/layer.json',url,f='json')
    reference='2026-06' if version=='2023' else '2024-01'
    codes = sorted(pd.read_parquet(OUT/f'consents/auckland_sa2_{version}_{reference}.parquet',columns=['SA2_code']).SA2_code.unique())
    features=[]
    for start in range(0,len(codes),50):
        batch=codes[start:start+50]
        where=field+' IN ('+','.join("'"+c+"'" for c in batch)+')'
        d=sources.json(f'sa2/{version}/batch_{start:04}.geojson',url+'/query',f='geojson',where=where,outFields=f'OBJECTID,{field},{name}',returnGeometry='true',outSR='2193')
        if d.get('exceededTransferLimit') or sorted(x['properties'][field] for x in d['features'])!=batch:
            raise ValueError('Missing or duplicate SA2 response')
        features.extend(d['features'])
        print('SA2 batch',start,flush=True)
    g=gpd.GeoDataFrame.from_features(features,crs=2193).rename(columns={field:'sa2_code',name:'sa2_name'})
    repairs=[]
    g['geometry_repaired']=~g.is_valid
    for i in g.index[g.geometry_repaired]:
        before=g.loc[i,'geometry']; after=make_valid(before)
        repairs.append({'sa2_code':g.loc[i,'sa2_code'],'area_before_m2':before.area,'area_after_m2':after.area,'relative_area_change':abs(after.area-before.area)/before.area})
        g.loc[i,'geometry']=after
    write_json(ROOT/f'reports/jiayi_sa2_{version}_repairs.json',repairs)
    if not g.sa2_code.is_unique or not g.geometry.is_valid.all() or g.geometry.is_empty.any():
        raise ValueError('SA2 geometry/key quality failure')
    g.to_parquet(OUT/f'sa2_{version}_auckland.parquet',index=False)
    return g


def classify(points, polygons, key):
    hits=gpd.sjoin(points[['geometry']],polygons[[key,'geometry']],how='inner',predicate='intersects')
    groups=hits.groupby(level=0)[key].agg(lambda v:sorted(set(v)))
    result=pd.DataFrame(index=points.index)
    result['hits']=pd.Series([groups.get(i,[]) for i in points.index],index=points.index)
    result['boundary']=False
    for i,row in hits.iterrows():
        if polygons.loc[row.index_right].geometry.boundary.distance(points.loc[i].geometry)<1e-6:
            result.loc[i,'boundary']=True
    return result


def rolling_total(frame, end):
    window=frame.loc[frame.month.between(end-pd.DateOffset(months=11),end)]
    expected=pd.date_range(end-pd.DateOffset(months=11),end,freq='MS')
    if window.month.duplicated().any() or set(window.month)!=set(expected) or window.total_dwelling_units.isna().any():
        return pd.NA
    return int(window.total_dwelling_units.sum())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    releases=pd.read_parquet(OUT/'consents_release_index.parquet').sort_values('earliest_usable_sale_date')
    for i,r in releases.loc[releases.sa2_version.eq('2024')].iterrows():
        path=OUT/f'consents/auckland_sa2_2018_{r.release_id}.parquet'
        if not path.exists():
            normalize_release(json.loads((RAW/f'consents/{r.release_id}/release_metadata.json').read_text(encoding='utf-8')),preferred_version='2018')
        releases.loc[i,'sa2_version']='2018'
    releases[['release_id','sa2_version','published_at_nz','earliest_usable_sale_date']].to_parquet(OUT/'consents_linkage_release_index.parquet',index=False)
    totals=[]
    for _,r in releases.iterrows():
        frame=pd.read_parquet(OUT/f'consents/auckland_sa2_{r.sa2_version}_{r.release_id}.parquet')
        end=pd.Timestamp(r.release_id+'-01')
        window=frame.loc[frame.month.between(end-pd.DateOffset(months=11),end)]
        for code,group in window.groupby('SA2_code'):
            totals.append({'release_id':r.release_id,'sa2_version':r.sa2_version,'sa2_code':code,'new_dwellings_consented_12m':rolling_total(group,end),'published_at_nz':r.published_at_nz,'earliest_usable_sale_date':r.earliest_usable_sale_date,'months_available':group.month.nunique()})
    total_frame=pd.DataFrame(totals)
    total_frame['new_dwellings_consented_12m']=total_frame.new_dwellings_consented_12m.astype('Int64')
    total_frame.to_parquet(OUT/'consents_12m_by_release_sa2.parquet',index=False)
    sources=Sources(args.offline)
    boundaries={v:collect(sources,v) for v in ['2023','2018']}
    sa2=boundaries['2023']
    noise=gpd.read_parquet(OUT/'aircraft_noise_operative.parquet')
    inputs=pd.read_parquet(OUT/'property_input.parquet')
    inputs=inputs.loc[inputs.coordinate_valid].copy()
    if inputs.empty:
        raise ValueError('No valid coordinates for the independent spatial sample.')
    sample=inputs.sample(n=min(200,len(inputs)),random_state=760).sort_values('source_index').reset_index(drop=True)
    points=gpd.GeoDataFrame(sample,geometry=gpd.points_from_xy(sample.longitude,sample.latitude),crs=4326).to_crs(2193)
    sh=classify(points,sa2,'sa2_code')
    spatial={v:classify(points,g,'sa2_code') for v,g in boundaries.items()}
    nh=classify(points,noise,'TYPE')
    sample['sa2_2023_hits']=sh.hits.map(json.dumps)
    sample['sa2_2023_status']=['boundary_ambiguous' if b else 'matched' if len(h)==1 else 'outside_or_multiple' for h,b in zip(sh.hits,sh.boundary)]
    sample['noise_types_raw']=nh.hits.map(json.dumps)
    sample['noise_match_status']=['boundary_ambiguous' if b else 'matched_current_snapshot' if h else 'outside_current_overlay' for h,b in zip(nh.hits,nh.boundary)]
    # Exact category combinations preserve overlaps; no unsupported dB conversion/ranking.
    labels=noise.set_index('TYPE').type_label.to_dict()
    sample['noise_category_combination']=nh.hits.map(lambda hits:' | '.join(labels[x] for x in hits) if hits else 'outside_current_overlay')
    sample.loc[nh.boundary,'noise_category_combination']=pd.NA
    sample['noise_temporal_status']='current_snapshot_historical_validity_unconfirmed'
    releases['available']=pd.to_datetime(releases.earliest_usable_sale_date)
    tables={}
    records=[]
    for i,row in sample.iterrows():
        available=releases.loc[releases.available.le(row.sale_date)]
        rec={'consents_match_status':'no_prior_release','sa2_new_dwellings_consented_12m':pd.NA}
        if not available.empty:
            release=available.iloc[-1]
            rec.update(consents_release_id=release.release_id,consents_geography_version=release.sa2_version,consents_published_at=release.published_at_nz)
            version=str(release.sa2_version)
            match=spatial[version].loc[i]
            if match.boundary or len(match.hits)!=1:
                rec['consents_match_status']='sa2_unresolved'
            else:
                rid=release.release_id
                if rid not in tables:
                    tables[rid]=pd.read_parquet(OUT/f'consents/auckland_sa2_{version}_{rid}.parquet')
                code=match.hits[0]
                value=rolling_total(tables[rid].loc[tables[rid].SA2_code.eq(code)],pd.Timestamp(rid+'-01'))
                rec.update(consents_sa2_code=code,sa2_new_dwellings_consented_12m=value,consents_match_status='matched_release_specific_snapshot' if pd.notna(value) else 'incomplete_window')
                area=boundaries[version].set_index('sa2_code').loc[code]
                rec['consents_geometry_repaired']=bool(area.geometry_repaired)
                rec['consents_boundary_distance_m']=points.loc[i].geometry.distance(area.geometry.boundary)
                if area.geometry_repaired:
                    rec['consents_match_status']='geometry_repair_requires_review'
                    rec['sa2_new_dwellings_consented_12m']=pd.NA
        records.append(rec)
    sample=pd.concat([sample,pd.DataFrame(records)],axis=1)
    sample['sa2_new_dwellings_consented_12m']=sample.sa2_new_dwellings_consented_12m.astype('Int64')
    assert len(sample)==min(200,len(inputs)) and sample.record_id.is_unique
    used=sample.consents_published_at.notna()
    assert (pd.to_datetime(sample.loc[used,'consents_published_at']).dt.normalize()<sample.loc[used,'sale_date']).all()
    sample.to_parquet(OUT/'spatial_sample_200.parquet',index=False)
    summary={'sample_seed':760,'sample_rows':len(sample),'sa2_polygons':len(sa2),'sa2_status':sample.sa2_2023_status.value_counts().to_dict(),'noise_status':sample.noise_match_status.value_counts().to_dict(),'noise_combinations':sample.noise_category_combination.value_counts().to_dict(),'consents_status':sample.consents_match_status.value_counts().to_dict(),'historical_noise_verified':False,'manual_address_accuracy_verified':False,'model_training_performed':False}
    write_json(ROOT/'reports/jiayi_spatial_sample_quality.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
