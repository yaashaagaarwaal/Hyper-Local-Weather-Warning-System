# India synthetic datasets (flood, lightning)

**Everything in this folder is a computer simulation.** No file here is a satellite
observation, a recorded flood, or detected lightning. Each sample's metadata carries
`"synthetic": true`, and the app shows these samples with a "Synthetic simulation" tag.
They exist so the flood and lightning pipelines can be developed and demonstrated on
Indian places, seasons and weather types while real Indian ground truth is unavailable.

Measured result (see `docs/flood_lightning_model_report.md`): training on these samples
did **not** improve accuracy on real data. Use them for development, demos and tests —
not as evidence of real-world performance.

## Regenerate

The arrays (`chips/`, `samples/`) are gitignored and fully reproducible (fixed seed 26077):

```bash
cd backend && source .venv/bin/activate
python -m ml.hazards.synth_flood        # -> dataset/synthetic/india_flood/      (~2 min, 208 MB)
python -m ml.hazards.lightning_prepare  # once: the GOES statistic the lightning generator borrows
python -m ml.hazards.synth_lightning    # -> dataset/synthetic/india_lightning/  (~20 s, 21 MB)
```

`synth_flood` needs the real Sen1Floods11 India training chips (brightness calibration);
`synth_lightning` uses the INSAT-3DR files if present (pass `--no-insat` to skip them —
the 16 scenes that use a real background then become fully procedural).

## `india_flood/` — 240 simulated Sentinel-1 flood chips

| File | Content |
|---|---|
| `chips/<id>.npz` | `s1` float16 (2, 512, 512): VV, VH backscatter in dB · `label` int8: 1 water, 0 not water · `permanent` uint8: river / sea / pond · `landcover` uint8: simulator class |
| `metadata.jsonl` | One line per chip: state, region, place, latitude, longitude, bounds, month, season, scenario, flood kind, flood / no-flood label, water and flood fractions, flooded area, split, seed |
| `summary.json` | Counts by split, region, state, scenario |
| `calibration.json` | The shift/scale that matches brightness to the real India training chips, and what it could not match |
| `quicklooks/*.png` | Contact sheets per flood kind |

- 40 real Indian sites × 6 variants (5 flooded + 1 dry-season); split 160 / 40 / 40.
- Layout and label convention copy Sen1Floods11, so the same code reads both.
- The mask is exact because the scene is built from it; image and mask are pixel-aligned.
- Deliberately included confusions: dry sand bars, radar shadow and roads are dark like
  water but labelled not-water; flooded crops and flooded streets are bright but water.

## `india_lightning/` — 408 simulated thunderstorm scenes

| File | Content |
|---|---|
| `samples/<id>.npz` | `bt` float16 (4, 128, 128): infrared cloud-top temperature (K), frames 30 min apart, 4 km pixels, row 0 = north · `flash_count` uint16 (4, 64, 64): simulated flashes per 8 km cell within ±5 min · `flashes` float32 (n, 4): frame, latitude, longitude, minute offset |
| `metadata.jsonl` | One line per scene: state, region, place, latitude, longitude, bounds, month, season, scenario, storm kind, lightning / no-lightning label, flashes per frame, coldest cloud top, split, seed, `foundation` |
| `summary.json`, `calibration.json`, `quicklooks/*.png` | As above |

- 34 real Indian sites × (8 storm + 4 storm-free variants); split 272 / 68 / 68.
- Lightning follows each storm cell's life cycle (peaks while the top is still growing,
  stops in decay), so cold cloud and lightning are not the same thing — as in real data.
- The flash-rate scale is borrowed from one GOES-16/GLM statistic measured over the
  **Americas**; it is an assumption for India, not a measurement.
- 16 western-disturbance scenes use four consecutive **real INSAT-3DR** infrared frames
  (Jan 2025, ISRO MOSDAC) as the cloud background. Their storm cells and all of their
  lightning are still simulated, and they are still tagged synthetic.
