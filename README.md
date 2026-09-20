# Explainable Auckland Property Price Prediction

COMPSCI 760 — Group 3, University of Auckland

We are investigating whether Auckland-specific information improves residential property-price predictions and using SHAP to explain how the models use that information. The project also aims to compare predictions with freely available valuation estimates where comparable observations are available.

## Current progress

This is a preliminary project update. The team has moved from a web-scraping approach to Auckland Council District Valuation Roll sales data (DVRS), prepared property and location fields, and used GLAM Geocoder to obtain coordinates for linking external datasets.

- **Property modelling:** Random Forest and XGBoost models using property attributes, location and capital value.
- **Existing external data:** flood plains, flood-sensitive areas, Unitary Plan zones and SA2 geography.
- **Accessibility:** walking and driving road-network distances to supermarkets and bus stops, using OpenStreetMap roads and supermarket locations and Auckland Transport bus-stop data.
- **Noise and development:** aircraft noise planning zones and SA2 residential building consents over the preceding 12 months. Consent features use releases available before each sale.
- **Other investigations:** a construction-material/building-era proxy for leaky-home risk, alternative Council–LINZ record linkage using Splink, and comparisons between older and more recent sales. These are separate investigations and are not all included in the results below. The leaky-home indicator is a proxy, not a diagnosis of building defects.

## Preliminary results

Early experiments suggest that additional contextual data provide limited improvements beyond the existing property and location inputs. Capital value dominates the reported SHAP results. Thao also tested removing capital value and observed poorer prediction accuracy; its timing still needs to be checked before interpreting this as evidence for prospective use.

The following is **Jay and Jiayi's rerun using Dominik's proposed filters**, with 47,617 training records and 8,324 test records. Both configurations use the same records, preprocessing and model parameters, without hyperparameter tuning. The baseline already includes Thao's existing flood and zoning inputs.

| Configuration | RF RMSE (NZD) | RF R² | XGBoost RMSE (NZD) | XGBoost R² |
|---|---:|---:|---:|---:|
| Existing inputs | 133,564 | 0.9386 | 131,293 | 0.9407 |
| + Travel distances, noise and residential consents | 133,466 | 0.9387 | 130,377 | 0.9415 |

RMSE improved slightly for both models. RF MAE increased from NZD 87,234 to 87,348, while XGBoost MAE decreased from NZD 86,406 to 86,259. The results therefore show small, metric-dependent changes rather than a consistent improvement across every measure.

These figures belong to this rerun. They should not be mixed with rows from other team experiments: the RF baseline matches the presentation's existing-external-data result to displayed precision, but the XGBoost baseline differs. Exact replication of all presentation results remains unresolved.

### Filters and evaluation

The rerun applies the five filters proposed by Dominik:

- First character of `zoning` is `9`, `1` or `2` (residential, rural or lifestyle).
- `units_of_use = 1`.
- `sale_type = S`.
- `sale_tenure = 1` (freehold).
- `price_value_relationship = 1` (market transactions).

It also retains Thao's `street_in_linz` check. The notebook's chronological split is recomputed after filtering: training ends on 11 February 2026 and testing begins on 12 February 2026. Training-derived 1st–99th percentile price bounds are also applied to test sale prices, following the notebook, so the metrics describe that restricted cohort rather than all Auckland sales.

## Data sources and code

| Source | Use |
|---|---|
| Auckland Council DVRS | Sale prices and property information |
| Auckland Council spatial datasets | Flood exposure, planning zones and aircraft noise zones |
| [OpenStreetMap](https://www.openstreetmap.org/) / [Geofabrik NZ extract](https://download.geofabrik.de/australia-oceania/new-zealand.html) | Road networks and supermarket locations |
| [Auckland Transport GTFS](https://gtfs.at.govt.nz/gtfs.zip) | Bus-stop locations |
| Stats NZ | Statistical geography and residential building consent data |

Travel distances are measured in metres along walking/driving networks, not travel time. Noise variables represent mapped categories, not continuous decibel measurements. Current road, facility and noise snapshots support retrospective analysis and are not verified historical snapshots for every sale.

Jay's calculation code and setup instructions are available on the [`xyan824` branch](https://github.com/hugn456/COMPSCI_760_Group_3_Intelligent_Real_Estate_Price_Prediction_Model/tree/xyan824/xyan824). Generated distance records retain `source_index` to join back to the original sales table without relying on row order. The branch contains code and documentation only; data and generated outputs are shared separately within the team.

## Next steps

- Agree on shared filters, feature definitions, preprocessing, splits and model settings for the final comparison.
- Link DVRS sales with DVRP valuation information, check the identifier and join cardinality, and establish whether capital values were available before each sale. Post-sale valuation information could introduce leakage; this has not yet been ruled out.
- Use time-based cross-validation within development data to compare feature combinations and tune hyperparameters. Keep a separate final evaluation period out of those choices; the current test results have already been inspected repeatedly.
- Investigate better external-data representations and two-stage residual modelling. For the latter, residual construction and evaluation should avoid reusing held-out outcomes.
- Continue investigating record linkage, geocoding quality and changes across sales periods.
- Assess whether prospective service estimates and subsequent sale outcomes can support the external valuation comparison within the project timeframe.

## Team contributions

| Member | Work to date |
|---|---|
| Huy Nguyen | Presentation coordination and integration, collaborating on the methodology and results update |
| Dominik Wecke | Council data transformations, modelling with existing property data in R, filtering proposals, construction-risk proxy, investigations of sales distributions and probabilistic record linkage, and presentation contributions |
| Thao Nguyen | GLAM geocoding, external flood and planning-data linkage, Random Forest/XGBoost modelling, SHAP analysis and capital-value experiments |
| Jiayi Du (Wendy) | Aircraft noise and residential consent data preparation, spatial and temporal alignment with property records, and presentation contributions |
| Xiang Yang (Jay) | Walking/driving network-distance calculation, additional-data comparisons, reruns with the proposed filters, and presentation contributions |

The literature survey and preliminary presentation are separate deliverables. This README summarises project progress; it does not claim that the final evaluation or all proposed experiments are complete.
