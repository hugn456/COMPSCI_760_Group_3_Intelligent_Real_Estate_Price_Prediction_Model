"""Read-only source adapter adapted from the existing shared Council loader."""
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_jiayi_data import digest


def load_council(path: Path) -> tuple[pd.DataFrame, dict]:
    source = pd.read_parquet(path)
    missing = {"longitude", "latitude", "sale_date"}.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not source.index.is_unique:
        raise ValueError("Source index is not unique; a reviewed row mapping is required")
    if not pd.api.types.is_integer_dtype(source.index.dtype):
        raise ValueError("Expected the team's original integer Parquet index")
    source_hash = digest(path)
    frame = pd.DataFrame({"source_index": source.index.to_numpy()})
    frame["source_sha256"] = source_hash
    frame["record_id"] = source_hash + ":" + frame.source_index.astype(str)
    for col in ("longitude", "latitude", "geocode_confidence"):
        frame[col] = pd.to_numeric(source[col] if col in source else pd.Series(np.nan, index=source.index), errors="coerce").to_numpy(dtype=float)
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
        "path": str(path.resolve()), "sha256": source_hash, "rows": len(source),
        "columns": len(source.columns), "exact_duplicate_rows": int(source.duplicated().sum()),
        "invalid_coordinates": int((~frame.coordinate_valid).sum()),
        "unparsed_sale_dates": int(frame.sale_date.isna().sum()),
        "sale_date_min": str(frame.sale_date.min()), "sale_date_max": str(frame.sale_date.max()),
        "geocode_confidence_below_0_6": int((frame.geocode_confidence < .6).sum()),
        "row_policy": "All rows retained, including duplicates; join by source_index after checking source_sha256",
    }
    return frame, metadata
