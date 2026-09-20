"""Read-only Council adapter: preserve every source row and its original index."""
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file


def load_council(path: Path) -> tuple[pd.DataFrame, dict]:
    source = pd.read_parquet(path)
    if not source.index.is_unique:
        raise ValueError("Source index is not unique; a reviewed row mapping is required")
    if not pd.api.types.is_integer_dtype(source.index.dtype):
        raise ValueError("Expected the team's original integer Parquet index")
    digest = sha256_file(path)
    frame = pd.DataFrame({"source_index": source.index.to_numpy()})
    frame["source_sha256"] = digest
    frame["record_id"] = digest + ":" + frame.source_index.astype(str)
    for col in ("longitude", "latitude", "geocode_confidence"):
        frame[col] = pd.to_numeric(source[col], errors="coerce").to_numpy(dtype=float)
    frame["sale_date"] = pd.to_datetime(
        source.sale_date.astype("string").str.strip().str.zfill(8),
        format="%d%m%Y", errors="coerce",
    ).to_numpy()
    frame["coordinate_valid"] = (
        np.isfinite(frame.longitude) & np.isfinite(frame.latitude)
        & frame.longitude.between(-180, 180) & frame.latitude.between(-90, 90)
        & ~((frame.longitude == 0) & (frame.latitude == 0))
    )
    metadata = {
        "path": str(path.resolve()), "sha256": digest, "rows": len(source),
        "columns": len(source.columns), "exact_duplicate_rows": int(source.duplicated().sum()),
        "invalid_coordinates": int((~frame.coordinate_valid).sum()),
        "unparsed_sale_dates": int(frame.sale_date.isna().sum()),
        "sale_date_min": str(frame.sale_date.min()), "sale_date_max": str(frame.sale_date.max()),
        "geocode_confidence_below_0_6": int((frame.geocode_confidence < .6).sum()),
        "row_policy": "All rows retained, including duplicates; join by source_index after checking source_sha256",
    }
    return frame, metadata


def merge_travel_features(source: pd.DataFrame, features: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Validated one-to-one join to the exact original Parquet, including subsets."""
    digest = sha256_file(source_path)
    if set(features.source_sha256.dropna()) != {digest} or features.source_sha256.isna().any():
        raise ValueError("Feature snapshot does not match the input file SHA256")
    if not source.index.is_unique or features.source_index.duplicated().any():
        raise ValueError("Merge keys must be unique")
    # Also check the caller's in-memory source was not reset/reordered into new keys.
    original = pd.read_parquet(source_path)
    if not source.index.isin(original.index).all() or not source.equals(original.loc[source.index]):
        raise ValueError("Pass original source rows with their original index, before model preprocessing")
    if not features.source_index.isin(original.index).all():
        raise ValueError("Unknown source indices in feature table")
    cols = [c for c in features if c.endswith(("_network_m", "_euclidean_m", "_status", "_distance_band", "_snap_m", "_boundary_sensitive"))
            or "_within_" in c or c in {"record_id", "source_sha256", "source_index"}]
    return source.join(features[cols].set_index("source_index"), how="left", validate="one_to_one")
