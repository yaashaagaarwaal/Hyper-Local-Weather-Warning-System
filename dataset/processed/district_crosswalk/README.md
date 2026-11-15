# geoBoundaries <-> LGD district crosswalk

- **Built by:** `dataset/scripts/build_district_crosswalk.py`
- **Method:** point-in-polygon spatial join of each geoBoundaries ADM2
  district's centroid against geoBoundaries' own ADM1 state layer (same
  source/provider), then an exact (name, state) join against the
  authoritative LGD district table. Only 2 documented, geometry-scoped
  renames are applied (see VERIFIED_RENAMES in the script) -- nothing else
  is guessed.
- **Match status counts:** {'matched_exact': 608, 'unmatched': 117, 'matched_name_unique_spatial_conflict': 7, 'matched_renamed': 3}
- **Spatial-join-uncertain rows:** 2 (centroid fell outside
  every ADM1 polygon; nearest-centroid fallback used -- these rows are
  flagged `spatial_join_uncertain=True` in the CSV, not silently accepted)
- **Output:** `geoboundaries_lgd_crosswalk.csv` (735 rows)

## What `unmatched` means here

A district name+assigned-state pair that doesn't exist in the current LGD
table under that name, and isn't one of the 2 documented renames. Causes
observed during Phase 4 investigation: Karnataka/other-state spelling
variants (e.g. Bengaluru vs Bangalore), and districts created by splits
since geoBoundaries' Jan 2023 vintage. Each `unmatched` row still has its
spatially-assigned state recorded, for manual review -- never dropped.

## `matched_name_unique_spatial_conflict` (7 rows) — a real spatial-join limitation found during validation

For these 7 districts, the point-in-polygon state assignment DISAGREED with
the district's own name match in LGD (which is globally unique, i.e. not
one of the genuine 3 multi-state duplicates). The LGD name match was
trusted; the spatial assignment was wrong in every case checked:

- `Gautam Buddha Nagar` (correctly UP) — spatial join said Haryana.
- `Jhansi` (correctly UP) — spatial join said Madhya Pradesh.
- `Nandurbar` (correctly Maharashtra) — spatial join said Gujarat.
- `Daman`, `Diu` (correctly the merged UT) — spatial join said the old
  pre-merger ADM1 name.
- `Balrampur` (correctly UP) — spatial join said Chhattisgarh. **This also
  resolves the "2 Balrampur polygons" question from the prior
  investigation: one geoBoundaries polygon is UP's real Balrampur; the
  other is a mislabeled/stray polygon near Chhattisgarh, not a second
  genuine administrative unit.**
- `Raigarh` (correctly Chhattisgarh) — spatial join said Maharashtra. **Same
  resolution for the "2 Raigarh polygons" question: the Maharashtra one is
  a geoBoundaries artifact, not a real second Raigarh.**

**Conclusion: geoBoundaries' ADM1 state polygons are imprecise at some
state-border districts** (not limited to Delhi-NCR, though Gautam Buddha
Nagar — a priority NCR district — is one of them). The safety net above
catches this only when the district's name is globally unique in LGD; it
cannot catch a similar error for one of the 3 genuinely duplicate-named
districts (Bilaspur/Hamirpur/Pratapgarh), since there both a correct and an
incorrect answer are plausible by name alone. Those 3 remain dependent on
the spatial join being accurate, which this finding shows cannot be fully
trusted.
