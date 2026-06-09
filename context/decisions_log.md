# Decisions log

Chronological. Each entry: what was observed, what was decided, why. Deliverables
may only use methods traceable to an entry here.

## 2026-06-08

**D1 — Spectrograms are the working representation.**
Observed: the provided MATLAB and the course framing both center on
micro-Doppler spectrograms; the verified chain produces a clean, class-distinct
walking signature (`sandbox/dsp_check.png`). Decided: classify from
spectrograms. Why: it is the established representation for this dataset and the
methods the course teaches (STFT / time-frequency).

**D2 — Python port of the DSP matches MATLAB.**
Observed: reproduced range-time and micro-Doppler outputs consistent with the
MATLAB example on a walking file. Decided: use the Python chain in
`dsp_pipeline.md` as the fixed preprocessing. Why: lets the whole project stay
in notebooks/Python while faithfully following the supplied processing.

**D3 — Subject-independent evaluation, keyed on (dataset, subject).**
Observed: subject IDs repeat across datasets and refer to different people
(`P03`, `P08`). Decided: all train/test splitting holds out whole people, keyed
on (dataset, subject). Why: a random split leaks the same person into train and
test and inflates accuracy; this is the central methodological point for the
"critical attitude" grading axis.

**D4 — Follow the README label mapping.**
Observed: README datasheet and `Label_extract4.m` disagree on activities 4/5.
Decided: use README (A04 pickup, A05 drink). Why: the course calls the README
authoritative. Action: flag the inconsistency in the write-up.

**D5 — Stage the datasets.**
Observed: 7 datasets, varied sites/ages, one (NG Homes) without the fall class.
Decided: build and validate on December 2017 first, then scale via a config
list. Why: prove correctness cheaply before paying full preprocessing cost, and
turn the site/age variation into an honest generalization test rather than an
assumption.

## 2026-06-09

**D6 — Activity label comes from the `Ayy` field; 3 leading-digit typos found.**
Observed: `sandbox/index_files.py` parses all 1754 files. In 1751 the leading
digit `K` equals the `Ayy` activity; 3 disagree: `2P34A03R1` (ds3, June 2017),
`3P03A02R1` (ds2, March 2017), `6P01A05R03` (ds5, Feb 2019 UoG). Decided: take
the activity from the `Ayy` field. Why: `Label_extract4.m` (the provided label
tool) reads the activity from between `A` and `R`, so `Ayy` is the official
label field; the leading digit is a redundant copy with 3 typos. Flag in
write-up.

**D7 — Fall is the minority class; its dataset coverage differs from the brief.**
Observed: per-class counts are walk 312, sit 312, stand 311, pick 311,
drink 311, fall 197. Fall is absent from dataset 7 (West Cumbria, 0 files) and
nearly absent from dataset 6 (NG Homes, only subject P08, 3 files). This is the
reverse of the brief's note (which said ds6 lacks fall and ds7 has it). Decided:
treat fall as a 6th class where present; for any train/test pairing involving
ds6 or ds7, handle the missing fall explicitly (5-class subset). Why: the elderly
care sites simply did not record falls at scale, which is itself a reportable
generalization limitation. Dec 2017 (ds1) is perfectly balanced (60 x 6 = 360).

**D8 — Splitting unit is the (dataset, subject) pair.**
Observed: 106 distinct (dataset, subject) people; 34 bare subject IDs appear in
more than one dataset (e.g. P08 in ds5 and ds6; P57 in ds1 and ds4). Decided:
all grouping/splitting keys on `d{dataset}_s{subject}`. Why: confirms D3 with the
real numbers; a bare-ID split would leak people across folds.

**D9 — Model input: ±6 m/s crop, 128x128, per-image 40 dB normalization.**
Observed: `sandbox/per_activity_ds1.png` shows every class signature sits within
+-6 m/s; outside that band is empty. Recording durations differ by activity
(walking ~10 s, transient activities ~4.5 s). Decided: crop to |v|<=6 m/s,
resample to a fixed 128x128 image (linear magnitude cached), and at model time
convert to dB referenced to the per-image max, clipped to a 40 dB range and
scaled to [0,1]. Why: the crop drops empty Doppler bands; the fixed time width
normalizes recording-length differences so the model learns the signature, not
the protocol; per-image dB removes absolute-gain differences between recordings
and matches the MATLAB display convention. Same input feeds both the CNN and the
classical feature extractor for a fair comparison.

**D10 — The fixed range-bin window 10..30 is valid (dataset 1).**
Observed: `sandbox/range_window_ds1.png`, range-bin energy across 40 random ds1
files peaks at bin 13 with 73% of the energy inside bins 9..29. Decided: keep the
window from the MATLAB example. Why: the human return falls inside it; re-check
before trusting it on the elderly-care datasets where standoff range may differ.

**D11 — Two models: classical features + SVM, and a transfer-learned CNN.**
Observed (dataset 1, 5-fold subject-independent CV): physical-feature SVM
0.961 +- 0.014, RandomForest 0.939 +- 0.038, ResNet18 (1-channel, fine-tuned on
MPS) 0.961 +- 0.033. The SVM matches the CNN at lower variance, ~1 s training vs
~50 s, and stays interpretable. Decided: report both. The classical model is the
primary result for interpretability and cost; the CNN is the comparison and the
candidate for the generalization test. Why: an honest, like-for-like comparison
is the originality/critical-attitude payload, and on clean single-site data the
handcrafted micro-Doppler features are already sufficient.

**D12 — Confusion structure is physical; the dominant error is pick vs drink.**
Observed: both models confuse pick (recall ~0.88-0.92) mostly with drink and sit;
walking is ~perfect; the sit/stand mirror the brief predicted is largely resolved.
Decided: attribute pick/drink confusion to both being low-velocity arm/torso
motions with similar Doppler extent, and credit the resolved sit/stand pair to the
directional (positive vs negative Doppler) features. Why: ties conclusions to the
physics, the interpretation grading axis.

**D13 — The fixed range window fits the lab data but clips the care-home data.**
Observed (`sandbox/range_window_all.py`, 25 files/dataset): energy fraction in
bins 9..29 is 0.68-0.75 for datasets 1-5 (peaks at bins 12-17), but only 0.39
(ds6 NG Homes) and 0.54 (ds7 West Cumbria), peaking at bins 8-9. The care-home
subjects stand closer to the radar, so part of their return falls below the
window. Decided: keep the fixed lab window as the documented pipeline and report
this as a measured cause of the generalization gap (a geometry shift on top of
the age/site shift). Why: faithful to the provided method and turns a limitation
into evidence-backed analysis.

**D14 — Headline: lab->care transfer drops to ~0.77, CNN no better than SVM.**
Observed (5-class, fall dropped; source = ds1-5 lab/university 970 files / 66
subjects, target = ds6-7 care 587 files / 40 subjects): within-source
subject-independent CV SVM 0.891 +- 0.022; source->target SVM 0.767, CNN 0.777;
per-dataset SVM 0.789 (NG Homes), 0.744 (West Cumbria). The gap concentrates in
pick<->drink (e.g. SVM: 52 pick scored as drink, 31 drink as pick) and some
walk->sit. Decided: this is the reported headline. Why: (a) a 12-19 point gap is
an honest measure of cross-site/age generalization; (b) the extra capacity of the
CNN gives no cross-domain advantage over interpretable features, a notable result;
(c) the pick/drink collapse fits slower, smaller elderly arm motions, and the
range-window geometry shift (D13) removes part of the care-home return - both
physical, both evidenced.

## Open questions — all resolved before the clean build
- Final model input and normalization -> D9 (crop +-6 m/s, 128x128, per-image
  40 dB).
- Whether the fixed range-bin window holds across datasets -> D10 (yes for ds1-5),
  D13 (clips ds6-7).
- Classical vs CNN vs both -> D11 (both; SVM primary, CNN comparison).
- Per-class counts and imbalance -> D7 (fall is the minority, absent in ds7).
