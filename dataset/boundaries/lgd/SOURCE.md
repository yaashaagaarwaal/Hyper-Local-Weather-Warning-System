# LGD (Local Government Directory) reference tables

- **Provider:** community mirror of LGD's own exports, https://ramseraph.github.io/opendata/lgd
  (lgdirectory.gov.in itself has no public API -- confirmed during Phase 4 research)
- **Underlying authority:** Government of India, Ministry of Panchayati Raj --
  Local Government Directory (https://lgdirectory.gov.in)
- **Downloaded:** 2026-11-14T17:53:19.456326+00:00
- **Downloaded by:** `dataset/scripts/fetch_lgd_reference.py`
- **Districts:** 790 rows, columns: ['district_code', 'district_name', 'state_code', 'state_name', 'district_census_2001_code', 'district_census_2011_code']
- **States:** 36 rows, columns: ['state_code', 'state_version', 'state_name', 'state_name_local', 'state_census_2001_code', 'state_census_2011_code', 'state_or_ut']

## Use

Reference/lookup only -- joins geoBoundaries' `shapeName` and the India Flood
Inventory's district names to an authoritative (district_code, state_code,
state_name) triple. See `dataset/processed/district_crosswalk/` for the
actual crosswalk built from this table.

## Known limitation

This mirror is a third-party convenience copy, not the government portal
itself -- treat `district_code`/`state_code` as authoritative LGD identifiers,
but re-verify against lgdirectory.gov.in directly for anything safety-critical.
