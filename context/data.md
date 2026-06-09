# Data

Source: University of Glasgow "INSHEP" radar activity dataset (researchdata.gla.ac.uk/848).
Radar: Ancortek FMCW, C-band carrier 5.8 GHz, bandwidth 400 MHz, chirp (sweep)
duration 1 ms, 128 ADC samples per sweep. Transmit/receive Yagi antennas.

## File format (`.dat`)
Plain text, one value per line, read as a long 1-D complex array.
- Lines 1-4 are a header: `[fc, Tsweep_ms, NTS, Bw]` =
  `[5.8e9, 1.0, 128, 400e6]` (constant across all files checked).
- Lines 5..end are the radar data as complex numbers in `a+bi` / `a-bi` text
  form. In Python: read the file, replace `i`->`j`, parse with
  `np.asarray(values, dtype=np.complex128)`.

### What one complex value is, physically
Each value is one ADC sample of the **de-chirped beat-note** (the mixed-down
echo), carrying instantaneous amplitude and phase (I + jQ). A single value is
not meaningful alone. Structure appears after reshaping:
- 128 samples per sweep = **fast time** (within one chirp). An FFT along fast
  time converts beat frequency to **range** (distance to scatterers).
- The sequence of sweeps = **slow time** (chirp index ≈ real time, here
  1 sweep / ms → PRF = 1000 Hz). A second short-time FFT along slow time gives
  **Doppler / velocity** over time — the micro-Doppler signature used for
  classification.

A typical file: 1,280,000 data samples = 128 × 10,000 sweeps = **10 s** at
PRF 1000 Hz.

## Filename convention `KPxxAyyRz.dat`
- `K` (leading digit) and `Ayy` both encode the activity (1..6 / A01..A06).
- `Pxx` = subject ID within that dataset.
- `Rz` = repetition.

**Gotcha 1 — repetition zero-padding differs by dataset.** Dec 2017 and
Feb 2019 UoG use `R01/R02/R03`; the other five use `R1/R2/R3`. Parsing must
accept both (regex `R(\d+)` works).

**Gotcha 2 — subject IDs are not globally unique.** `Pxx` numbers restart per
dataset, so e.g. `P03` in March 2017 and `P03` in Feb 2019 UoG are different
people; `P08` appears in both NG Homes and Feb 2019 UoG. Any subject-independent
split must key on **(dataset, subject)**, never the bare `Pxx`, or it leaks the
same person across train/test.

**Gotcha 3 — label naming contradiction.** The README datasheet maps
`A04 = pick up`, `A05 = drink water`. The provided `Label_extract4.m` comment
maps `4 = drink water`, `5 = pick`. They disagree on positions 4 and 5. We
follow the README (the document the course calls authoritative). This only
matters for *naming* classes in the confusion-matrix discussion, not for
training. Flag it explicitly in the write-up as a noticed inconsistency.

## Per-dataset breakdown
| # | Folder | Files | Activities | Subjects | Notes |
|---|--------|------|-----------|----------|-------|
| 1 | 1 December 2017 | 360 | 6 (incl. fall) | P36–P56, 20, young males | lab, `R0x` padding |
| 2 | 2 March 2017 | 48 | 6 | P03,P10–P12, 4 | 2 reps only |
| 3 | 3 June 2017 | 162 | 6 | P14,P28–P35, 9 | mixed gender |
| 4 | 4 July 2018 | 288 | 6 | P57–P72, 16 | common room |
| 5 | 5 February 2019 UoG | 306 | 6 | P01–P17, 17 | lab, `R0x` padding |
| 6 | 6 February 2019 NG Homes | 301 | **5 (no fall)** | P08,P18–P36, 20 | elderly residents, 3 rooms |
| 7 | 7 March 2019 West Cumbria | 289 | 6 | P37–P56, 20 | Age UK centre, elderly |
Total: 1754 `.dat` files.

Age profile matters for the generalization story: datasets 1-4 are mostly young
adults; datasets 6-7 are older people in care settings (ages up to 98). A model
trained on the young/lab data and tested on the elderly/care data is a strong,
honest test of generalization.

## Recommended dataset staging
- **Stage A (prove it works):** dataset 1 (December 2017) only — clean, all 6
  classes, 20 subjects, single site.
- **Stage B (generalize):** add the rest via a config list. Decide which based
  on results, not upfront. Dataset 6 lacks the fall class — handle it explicitly
  (either drop fall from that comparison or keep it as a 5-class subset).
