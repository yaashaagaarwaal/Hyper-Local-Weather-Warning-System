# Delhi-NCR districts + India Flood Inventory stats (priority deliverable)

- **Built by:** `dataset/scripts/build_ncr_flood_districts.py`
- **Inputs (all already validated, none modified):** geoBoundaries ADM2
  geometry, the geoBoundaries<->LGD crosswalk, and the India Flood
  Inventory's 3 district-aggregate CSVs.
- **Districts included:** 20 (Delhi's sub-districts + the 10
  verified NCR-adjacent districts in Haryana/UP/Rajasthan).
- **Districts with real flood statistics:** 20/20.
- **Outputs:** `ncr_districts_with_flood_stats.geojson`,
  `ncr_districts_with_flood_stats.csv`.

## Correction to an earlier Phase 4 finding

The Stage-1 investigation checked only the literal string "Delhi" against
the flood aggregate files and found nothing, concluding Delhi had no flood
stats. That was checking the wrong name: **Delhi's actual sub-district
names (`North`, `South`, `East`, `West`, `Central`, `New Delhi`, `North
East`, `North West`, `South East`, `South West`) DO each appear as exactly
one row** in `District_FloodedArea.csv`/`District_FloodImpact.csv`/`DFSI.csv`
-- confirmed directly, not assumed. All 20 NCR districts in this output
have real flood statistics. These generic names carry a residual risk of
collision with some other state's district of the same name, but none were
found to be ambiguous (no duplicate rows) when checked.
