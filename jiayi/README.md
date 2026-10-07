# Jiayi / Wendy: aircraft noise and residential building consents

Code-only handoff of the existing feature-processing pipeline. No property records,
coordinates, downloaded datasets, cached source responses, generated feature tables,
models or results are included. Runtime inputs and outputs stay in ignored folders.

This folder prepares and links two features:

- `noise_category_combination`: the aircraft noise planning-zone categories intersecting
  the property point. Overlapping categories are preserved; they are not converted into
  decibels or added together.
- `sa2_new_dwellings_consented_12m`: new residential dwelling units approved in the preceding
  twelve reference months, taken from the latest Stats NZ release available before sale
  and linked using that release's SA2 boundary version. These are approvals, not completed homes.

## Layout

- `scripts/prepare_jiayi_data.py`: collect/cache public noise and consent sources; normalize
  release tables; create property inputs from a locally supplied source.
- `scripts/check_jiayi_spatial_sample.py`: prepare SA2 2018/2023 boundaries, release-specific
  twelve-month totals, and an independent sample of up to 200 valid-coordinate rows.
- `scripts/build_jiayi_features.py`: link both features to every original source row.
- `scripts/property_input.py`: preserve original integer row indices and source-file provenance.
- `scripts/merge_features.py`: merge only the two features and their quality/provenance fields.
- `tests/`: synthetic offline checks for boundaries, overlaps, missing months, publication
  timing, coordinate updates and safe merging.

The handoff adapts the existing scripts to portable paths. Property preparation no longer
depends on Jay's generated travel table. The sample uses up to 200 valid-coordinate rows;
the full feature output still retains invalid-coordinate rows with missing values and flags.
There is no Splink record-linkage implementation or house-price model training in this folder.

## Local input and setup

Use Python 3.13; dependency versions match the environment used for this handoff.
From the repository root in PowerShell:

```powershell
git switch jiayi
cd jiayi
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
```

Place the clean team sales table at `data/private/dvrs_processed.parquet` locally, or pass
its path with `--input` to both property preparation and feature generation. Required
columns are `longitude`, `latitude` and `sale_date` in DDMMYYYY format. Optional
`geocode_confidence` is retained if supplied. The Parquet index must be the unique,
original integer row key; do not reset it after Splink matching or coordinate updates.
If the updated table has a different row-key/schema policy, agree the mapping before running.

## Prepare sources and build features

```powershell
& .venv/Scripts/python.exe scripts/prepare_jiayi_data.py --noise --property-input --input data/private/dvrs_processed.parquet --start 2023-10 --end 2026-06
& .venv/Scripts/python.exe scripts/check_jiayi_spatial_sample.py
& .venv/Scripts/python.exe scripts/build_jiayi_features.py --input data/private/dvrs_processed.parquet
& .venv/Scripts/python.exe scripts/merge_features.py --source data/private/dvrs_processed.parquet --features data/processed/jiayi/dvrs_noise_consents_features.parquet --output data/processed/dvrs_with_noise_consents.parquet
```

The dates above reproduce the original consent-release scope. The boundary preparation
currently uses the 2026-06 SA2 2023 table and 2024-01 SA2 2018 table to select Auckland
codes, so both reference releases must be included. Other sale ranges require checking
release coverage and boundary vintages before interpreting the results.

Public downloads are cached under `data/raw/jiayi/`, normalized inputs under
`data/interim/jiayi/`, and features under `data/processed/jiayi/`. The cache manifest is
`metadata/jiayi_sources.json`; quality summaries go to `reports/`. None are tracked.
For exact reproduction, privately reuse the agreed source cache together with its manifest.
Adding `--offline` to the first two commands verifies cached responses instead of downloading.
Current downloads may differ from the team's earlier snapshots.

## Rerun with updated coordinates

Use a separate clean input Parquet and retain the original row keys. Regenerate
`property_input.parquet`, then rerun the sample and full linkage. With the existing public
cache available, the first preparation step can be reduced to:

```powershell
& .venv/Scripts/python.exe scripts/prepare_jiayi_data.py --offline --property-input --input data/private/dvrs_updated_coordinates.parquet
& .venv/Scripts/python.exe scripts/check_jiayi_spatial_sample.py --offline
& .venv/Scripts/python.exe scripts/build_jiayi_features.py --input data/private/dvrs_updated_coordinates.parquet
& .venv/Scripts/python.exe scripts/merge_features.py --source data/private/dvrs_updated_coordinates.parquet --features data/processed/jiayi/dvrs_noise_consents_features.parquet --output data/processed/dvrs_updated_with_noise_consents.parquet
```

Runtime generated files are replaced by a rerun; retain previous outputs privately if needed.
Old-coordinate features cannot be merged into a new source: the source SHA256 and index
coverage must match exactly. A hash change requires rebuilding, not bypassing the check.

## Interpretation and quality flags

- Noise is a current planning-overlay exposure proxy, not measured noise at the sale date.
  Historical validity has not been established.
- Consent availability is based on the published release date. Date-only sales use the
  following day as the earliest usable sale date. A cached historical URL does not establish
  that its archive has never been revised.
- SA2 2023 and SA2 2018 are kept separate. Releases initially choosing a 2024 table use
  their included 2018 table for spatial linkage; identical codes do not imply identical boundaries.
- Outside, multiple-hit, boundary-ambiguous and repaired-geography cases retain review
  flags and missing consent values. Missing is not zero. Explicit zero counts are preserved.
- Coordinate validity checks numerical ranges; it does not establish geocoding accuracy.
- Repeated property sales and duplicate-content rows remain distinct by source index.

## Offline tests

```powershell
& .venv/Scripts/python.exe -m pytest tests -q
```

Tests create synthetic temporary files locally. They use no team data and make no network requests.

## Source references

- [Auckland Council aircraft noise overlay](https://services1.arcgis.com/n4yPwebTjJCmXB6W/arcgis/rest/services/Aircraft_Noise_Overlay/FeatureServer/0)
- [Stats NZ SA2 2023 boundaries](https://services2.arcgis.com/vKb0s8tBIA3bdocZ/ArcGIS/rest/services/Statistical_Area_2_2023/FeatureServer/0)
- [Stats NZ SA2 2018 boundaries](https://services2.arcgis.com/vKb0s8tBIA3bdocZ/ArcGIS/rest/services/Statistical_Area_2_2018/FeatureServer/0)
- [Stats NZ building consents issued](https://www.stats.govt.nz/indicators/building-consents-issued/)

Jiayi/Wendy's noise and residential-consent work is separate from Jay's accessibility
pipeline. This branch packages the shared project's existing processing code and its
required helpers for a code-only handoff.
