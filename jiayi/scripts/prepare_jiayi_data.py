"""Prepare public noise/consents source tables; does not join DVRS or train models."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/jiayi'
OUT = ROOT/'data/interim/jiayi'
META = ROOT/'metadata/jiayi_sources.json'
NOISE = 'https://services1.arcgis.com/n4yPwebTjJCmXB6W/arcgis/rest/services/Aircraft_Noise_Overlay/FeatureServer/0'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sources:
    def __init__(self, offline=False):
        self.offline = offline
        self.entries = json.loads(META.read_text(encoding='utf-8')) if META.exists() else {}
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'auckland-house-data/0.1 academic-source-preparation'
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504])))

    def get(self, relative, url, params=None):
        path = RAW/relative
        key = path.relative_to(ROOT).as_posix()
        requested = url + ('?'+urlencode(params) if params else '')
        if path.exists():
            if key not in self.entries or self.entries[key]['requested_url'] != requested or self.entries[key]['sha256'] != digest(path):
                raise ValueError(f'Cache mismatch: {path}')
            return path
        if self.offline:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.part')
        with self.session.get(url,params=params,timeout=(20,120),stream=True) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    f.write(chunk)
            temporary.replace(path)
            self.entries[key] = {'requested_url':requested,'response_url':response.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':digest(path),'bytes':path.stat().st_size,'last_modified':response.headers.get('Last-Modified'),'etag':response.headers.get('ETag')}
        write_json(META,self.entries)
        return path

    def json(self, relative, url, **params):
        data = json.loads(self.get(relative,url,params).read_text(encoding='utf-8'))
        if 'error' in data:
            raise RuntimeError(data['error'])
        return data


def collect_noise(sources):
    meta = sources.json('noise/layer.json',NOISE,f='json')
    ids = sorted(sources.json('noise/ids.json',NOISE+'/query',f='json',where='1=1',returnIdsOnly='true')['objectIds'])
    features = []
    for offset in range(0,len(ids),100):
        batch = ids[offset:offset+100]
        d = sources.json(f'noise/features_{offset:04}.geojson',NOISE+'/query',f='geojson',objectIds=','.join(map(str,batch)),outFields='*',returnGeometry='true',outSR='2193')
        assert not d.get('exceededTransferLimit',False)
        assert sorted(x['properties']['OBJECTID'] for x in d['features']) == batch
        features.extend(d['features'])
    g = gpd.GeoDataFrame.from_features(features,crs=2193)
    assert g.OBJECTID.is_unique and len(g)==len(ids)
    for field in ['TYPE','VERSIONSTATUS','VALIDATIONSTATE']:
        definition = next(f for f in meta['fields'] if f['name']==field)
        mapping = {v['code']:v['name'] for v in definition['domain']['codedValues']}
        g[field.lower()+'_label'] = g[field].map(mapping)
        assert g.loc[g[field].notna(),field.lower()+'_label'].notna().all()
    g['geometry_valid'] = g.geometry.is_valid
    g['geometry_empty'] = g.geometry.is_empty
    g['source_snapshot_sha256'] = digest(RAW/'noise/layer.json')
    g['temporal_status'] = 'current_planning_snapshot_not_historical'
    # Preserve every record; the downstream selection is explicit and reversible.
    g['selection_status'] = 'review_status'
    selected = g.VERSIONSTATUS.eq(4) & g.VALIDATIONSTATE.eq(3) & g.geometry_valid & ~g.geometry_empty
    g.loc[selected,'selection_status'] = 'operative_public_valid_geometry'
    OUT.mkdir(parents=True,exist_ok=True)
    g.to_parquet(OUT/'aircraft_noise_all.parquet',index=False)
    g.loc[selected].to_parquet(OUT/'aircraft_noise_operative.parquet',index=False)
    summary = {'records':len(g),'selected':int(selected.sum()),'invalid_geometry':int((~g.geometry_valid).sum()),'empty_geometry':int(g.geometry_empty.sum()),'types':g.type_label.value_counts().to_dict(),'status_combinations':g.groupby(['versionstatus_label','validationstate_label'],dropna=False).size().reset_index(name='records').to_dict('records'),'historical_validity_confirmed':False}
    write_json(ROOT/'reports/jiayi_noise_source_quality.json',summary)
    print('NOISE',json.dumps(summary,ensure_ascii=False),flush=True)


def collect_release(sources, month):
    date = pd.Timestamp(month+'-01')
    slug = date.strftime('%B-%Y').lower()
    url = 'https://www.stats.govt.nz/information-releases/building-consents-issued-'+slug+'/'
    page = sources.get(f'consents/{month}/release.html',url)
    text = html.unescape(page.read_text(encoding='utf-8')).replace('\\/','/')
    dates = re.findall(r'"PublicationDate":"([^"]+)"',text)
    links = re.findall(r'"DocumentLink":"([^"]+)"',text)
    links = sorted(set(x for x in links if 'new-dwellings-consented-by-statistical-area-2' in x and x.endswith('.zip')))
    if not dates or len(links)!=1:
        raise ValueError(f'Cannot resolve release metadata: {month}, dates={dates}, links={links}')
    path = sources.get(f'consents/{month}/sa2.zip','https://www.stats.govt.nz'+links[0])
    with zipfile.ZipFile(path) as z:
        members = [{'name':i.filename,'bytes':i.file_size} for i in z.infolist()]
    release = {'release_id':month,'reference_end_month':month,'published_at_nz':dates[0],'earliest_usable_sale_date':str((pd.Timestamp(dates[0]).normalize()+pd.Timedelta(days=1)).date()),'source_page':url,'source_zip':'https://www.stats.govt.nz'+links[0],'sha256':digest(path),'members':members,'historical_archive_verified':False}
    write_json(RAW/f'consents/{month}/release_metadata.json',release)
    print('RELEASE',json.dumps(release,ensure_ascii=False),flush=True)
    return release


def normalize_release(release, preferred_version=None):
    month = release['release_id']
    with zipfile.ZipFile(RAW/f'consents/{month}/sa2.zip') as archive:
        members = [n for n in archive.namelist() if re.search(r'by \d{4} statistical area 2', n) and n.endswith('.csv')]
        preferred = [n for n in members if f'{preferred_version or "2023"} statistical area 2' in n]
        if preferred_version and not preferred:
            raise ValueError(f'{month}: required version {preferred_version} missing')
        member = preferred[0] if preferred else max(members, key=lambda n:int(re.search(r'by (\d{4})',n).group(1)))
        version = re.search(r'by (\d{4})',member).group(1)
        pieces = []
        for chunk in pd.read_csv(archive.open(member), dtype=str, chunksize=100000):
            pieces.append(chunk.loc[chunk.territorial_authority.eq('Auckland')].copy())
    frame = pd.concat(pieces, ignore_index=True)
    if frame.empty or frame.duplicated(['SA2_code','month']).any():
        raise ValueError(f'{month}: empty Auckland data or duplicate keys')
    frame['month'] = pd.to_datetime(frame['month'], errors='raise')
    end = pd.Timestamp(month+'-01')
    if frame['month'].max() != end:
        raise ValueError(f'{month}: archive reference period mismatch')
    measures = ['total_dwelling_units','houses','apartments','retirement_village_units','townhouses_flats_units_other']
    missing = {}
    for col in measures:
        raw = frame[col]
        values = pd.to_numeric(raw, errors='coerce')
        missing[col] = raw[values.isna()].fillna('<empty>').value_counts().to_dict()
        if ((values.dropna() < 0) | (values.dropna() % 1 != 0)).any():
            raise ValueError(f'{month}: invalid count in {col}')
        frame[col] = values.astype('Int64')
    complete = frame[measures].notna().all(axis=1)
    mismatch = complete & frame.total_dwelling_units.ne(frame[measures[1:]].sum(axis=1))
    if mismatch.any():
        raise ValueError(f'{month}: dwelling components do not sum to total')
    frame['sa2_version'] = version
    for key in ['release_id','published_at_nz','earliest_usable_sale_date']:
        frame[key] = release[key]
    frame['source_zip_sha256'] = release['sha256']
    frame = frame.sort_values(['SA2_code','month']).reset_index(drop=True)
    folder = OUT/'consents'
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder/f'auckland_sa2_{version}_{month}.parquet', index=False)
    window = frame.loc[frame.month.between(end-pd.DateOffset(months=11),end)]
    if not window.groupby('SA2_code').size().eq(12).all() or window.SA2_code.nunique()!=frame.SA2_code.nunique():
        raise ValueError(f'{month}: incomplete 12-month observation window')
    summary = {**release,'sa2_version':version,'archive_member':member,'auckland_rows':len(frame),'sa2_count':frame.SA2_code.nunique(),'first_month':str(frame.month.min().date()),'last_month':str(frame.month.max().date()),'duplicate_keys':0,'component_mismatches':0,'missing_tokens':missing,'window_rows':len(window)}
    report_name = f'{month}_{preferred_version}' if preferred_version else month
    write_json(ROOT/f'reports/jiayi_consents/{report_name}.json',summary)
    print(f'NORMALIZED {month}: {len(frame)} rows, {frame.SA2_code.nunique()} SA2s',flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--noise',action='store_true')
    parser.add_argument('--property-input',action='store_true')
    parser.add_argument('--input',type=Path,default=ROOT/'data/private/dvrs_processed.parquet')
    parser.add_argument('--months',nargs='*',default=[])
    parser.add_argument('--start')
    parser.add_argument('--end')
    args=parser.parse_args()
    sources=Sources(args.offline)
    if args.property_input:
        from property_input import load_council
        frame, metadata = load_council(args.input.resolve())
        if frame.sale_date.isna().any():
            raise ValueError('Unparseable sale_date: expected DDMMYYYY. Resolve before linkage.')
        OUT.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(OUT/'property_input.parquet',index=False)
        metadata['source_input'] = str(args.input.resolve())
        write_json(ROOT/'reports/jiayi_property_input_quality.json',metadata)
    if args.noise:
        collect_noise(sources)
    months = args.months
    if args.start or args.end:
        if not (args.start and args.end):
            parser.error('--start and --end are required together')
        months += pd.period_range(args.start,args.end,freq='M').astype(str).tolist()
    summaries, failures = [], []
    for month in sorted(set(months)):
        try:
            summaries.append(normalize_release(collect_release(sources,month)))
        except Exception as exc:
            failures.append({'release_id':month,'error':str(exc)})
            print(f'FAILED {month}: {exc}',flush=True)
    if months:
        write_json(ROOT/'reports/jiayi_consents_source_quality.json',{'releases':summaries,'failures':failures})
        pd.DataFrame([{k:v for k,v in r.items() if k not in ['members','missing_tokens']} for r in summaries]).to_parquet(OUT/'consents_release_index.parquet',index=False)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
