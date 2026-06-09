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

## Open questions (resolve with evidence before the clean build)
- Final model input: full spectrogram vs velocity-cropped (±6 m/s), and the dB
  normalization scheme.
- Whether the fixed range-bin window (10..30) holds across datasets.
- Classical-features + classical-model vs fine-tuned CNN vs both — decide from
  measured accuracy and training cost, not upfront.
- Per-class counts and any class imbalance once labels are parsed for real.
