"""Distance helpers reused from Jay's existing accessibility pipeline."""
from typing import Iterable
import geopandas as gpd
import pandas as pd

def _threshold_label(km: float) -> str:
    if km < 1:
        return f"{int(round(km * 1000))}m"
    if float(km).is_integer():
        return f"{int(km)}km"
    return f"{km:g}km"

def distance_bands(distances_m: pd.Series, thresholds_km: Iterable[float]) -> pd.Series:
    thresholds = [float(value) for value in thresholds_km]
    if not thresholds or thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("Distance thresholds must be unique and sorted in ascending order")

    result = pd.Series(pd.NA, index=distances_m.index, dtype="string")
    available = distances_m.notna()
    previous_km: float | None = None
    for threshold_km in thresholds:
        unassigned = available & result.isna() & distances_m.le(threshold_km * 1000)
        if previous_km is None:
            label = f"within_{_threshold_label(threshold_km)}"
        else:
            label = f"{_threshold_label(previous_km)}_to_{_threshold_label(threshold_km)}"
        result.loc[unassigned] = label
        previous_km = threshold_km
    result.loc[available & result.isna()] = f"over_{_threshold_label(thresholds[-1])}"
    return result

def presence_flag(distances_m: pd.Series, threshold_km: float) -> pd.Series:
    result = pd.Series(pd.NA, index=distances_m.index, dtype="boolean")
    available = distances_m.notna()
    result.loc[available] = distances_m.loc[available].le(float(threshold_km) * 1000)
    return result

def _nearest_distances(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> pd.Series:
    if right.empty:
        return pd.Series(float("nan"), index=left.index, dtype="float64")
    if left.crs != right.crs:
        raise ValueError("Nearest-distance inputs must use the same CRS")
    joined = gpd.sjoin_nearest(
        left[["geometry"]],
        right[["geometry"]],
        how="left",
        distance_col="distance_m",
    )
    return joined.groupby(level=0)["distance_m"].min().reindex(left.index)
