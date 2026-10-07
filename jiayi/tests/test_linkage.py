import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_jiayi_spatial_sample import classify, rolling_total


def test_window_preserves_zero_but_rejects_missing_and_duplicate_months():
    months=pd.date_range('2023-01-01',periods=12,freq='MS')
    d=pd.DataFrame({'month':months,'total_dwelling_units':pd.array([0]*12,dtype='Int64')})
    assert rolling_total(d,months[-1])==0
    assert pd.isna(rolling_total(d.iloc[1:],months[-1]))
    assert pd.isna(rolling_total(pd.concat([d,d.iloc[:1]]),months[-1]))
    d.loc[3,'total_dwelling_units']=pd.NA
    assert pd.isna(rolling_total(d,months[-1]))


def test_spatial_overlap_boundary_and_outside_preserved():
    polygons=gpd.GeoDataFrame({'code':['outer','inner']},geometry=[box(0,0,10,10),box(2,2,8,8)],crs=2193)
    points=gpd.GeoDataFrame(geometry=[Point(5,5),Point(0,5),Point(20,20)],crs=2193)
    r=classify(points,polygons,'code')
    assert r.loc[0,'hits']==['inner','outer']
    assert bool(r.loc[1,'boundary'])
    assert r.loc[2,'hits']==[] and not r.loc[2,'boundary']
