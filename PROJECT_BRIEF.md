# Project Brief: Human Activity Classification with Radar

Single source of truth for this project. A fresh session should read this file
top to bottom before doing anything. It contains the task, the constraints, the
verified facts about the data, the proven signal processing recipe, the
decisions made so far with their justification, and the way of working. The
`context/` folder holds the same material split into topic files if that is
easier to navigate; this file is authoritative.

Everything below that is stated as verified was checked against the real data or
the provided code. Nothing here is assumed.

---

## 1. The task

Course EE4775, Object Classification with Radar (TU Delft, Dr. F. Fioranelli).
You are given radar recordings of people performing daily activities. Classify
each recording into one of six activities. The professor frames it as an open
problem: build a classification pipeline that performs as well as is reasonable
and justify every choice. Group of three.

Activities, using the README datasheet numbering (this is the authoritative
document):
1 walking, 2 sitting down, 3 standing up, 4 picking up an object, 5 drinking,
6 falling.

### Deliverables
- A one page description of the implementation and the ideas behind it.
- The code, runnable and reproducible, delivered as Jupyter notebooks in a
  `notebooks/` folder.
- The trained model(s), documented so they can be re-run and checked.
- A presentation of about nine minutes plus questions. Not submitted, but ready.

The one pager, code, and presentation together are worth 25% of the course mark.

### Grading, four axes at 25% each
1. Correct use of radar signal processing and ML methods, with clear reasons.
2. Interpretation of results and critical attitude (conclusions tied to results).
3. Originality of the proposed solution.
4. Presentation: structure, clarity, handling of questions.

Aim for a solid, defensible pass in the good band. The professor explicitly says
not to chase perfect accuracy but to produce reasonable solutions with
meaningful analysis.

---

## 2. How to work

Evidence first. Do not pre-commit to a full pipeline. The order is always:

1. Investigate one question in `sandbox/` with a short `.py` script. Save the
   output that answers it (a plot, a printed metric, a table).
2. Look at the result. Write down the finding and what it implies.
3. Let that finding decide the next step.
4. Only once a part of the approach is settled by evidence, build the clean
   notebook for it in `notebooks/`.

Two working areas:
- `sandbox/` holds throwaway `.py` scripts, trial plots, and scratch output.
  Nothing here is a deliverable. It is where decisions get made.
- `notebooks/` holds the professor facing build. Each notebook is produced only
  after the sandbox has proven the approach it documents.

Every method, number, and figure that appears in a notebook must trace back to a
finding in the sandbox and an entry in the decisions log (Section 7 here, or
`context/decisions_log.md`). If it is not backed, it does not go in.

---

## 3. Writing and documentation rules

These apply to all notebooks, the one pager, the presentation, and the markdown
in this project.

- Document thoroughly. Functions get docstrings that state what they do, the
  inputs and outputs, and any non obvious choice. Cells get a short markdown
  note explaining the why, not a narration of the code.
- Write tightly. Concise, direct, on point. Say the thing once. No filler, no
  hedging, no restating the obvious.
- Write like a competent engineer, not like a chatbot. No emojis. No em dashes
  or decorative dashes used as punctuation (ordinary hyphens inside technical
  terms such as micro-Doppler, slow-time, high-pass are fine). No marketing
  tone, no rhetorical questions, no "let's", no exclamation points.
- Keep terminology coherent with the course: FMCW, chirp, beat note, fast time,
  slow time, range FFT, MTI, STFT, spectrogram, micro-Doppler, Doppler,
  velocity. Reference the lecture material where it fits naturally, do not force
  it.
- A teammate should be able to read a notebook once before the presentation and
  defend every step from it.

---

## 4. Environment

- Apple silicon MacBook. Python 3.13 via Homebrew.
- Use a virtual environment. Required packages: numpy, scipy, scikit-learn,
  matplotlib, pillow, tqdm, torch, torchvision, jupyter, ipykernel.
- PyTorch MPS backend is available and working on this machine, so any CNN
  trains on the GPU in minutes. Confirmed: `torch.backends.mps.is_available()`
  returns True (torch 2.12, torchvision 0.27).
- Compute budget is minutes, not hours. Preprocess once and cache to disk.
  Classical models train in seconds. A fine-tuned CNN on MPS trains in a few
  minutes.

---

## 5. The data, and what it physically means

Source: University of Glasgow INSHEP radar activity dataset
(researchdata.gla.ac.uk/848). Radar: Ancortek FMCW, carrier 5.8 GHz, bandwidth
400 MHz, chirp (sweep) duration 1 ms, 128 ADC samples per sweep.

### File format (`.dat`)
Plain text, one value per line, read as a long 1-D complex array.
- Lines 1 to 4 are a header: `[fc, Tsweep_ms, NTS, Bw]` =
  `[5.8e9, 1.0, 128, 400e6]`, constant across every file checked.
- Lines 5 to end are the radar data as complex numbers in `a+bi` or `a-bi` text
  form. To parse in Python: read the file, replace `i` with `j`, then
  `np.asarray(values, dtype=np.complex128)`.

A typical file holds 1,280,000 data samples, which is 128 samples per sweep
times 10,000 sweeps, which is 10 seconds at a pulse repetition frequency of
1000 Hz.

### What the numbers mean
FMCW radar transmits a chirp, a wave whose frequency ramps linearly across the
400 MHz bandwidth during each 1 ms sweep. The echo from a target at range R
arrives delayed by the round trip time, so its frequency differs from the
currently transmitted frequency by a fixed amount, the beat frequency, which is
proportional to range:

    f_beat = (2 * B * R) / (Tsweep * c)

The receiver mixes the echo with the transmitted chirp and low-pass filters the
result, leaving the slow beat note. It samples that beat note 128 times per
chirp using I/Q (quadrature) detection, which records amplitude and phase, hence
each sample is complex. I/Q sampling is what lets the radar distinguish a target
approaching from one receding (positive vs negative Doppler).

So one complex value is one snapshot of the beat note, its instantaneous
strength and phase at one instant inside one chirp. It is meaningless alone.
Structure appears after reshaping the stream into a matrix with two time axes:

- 128 samples within a chirp are fast time. An FFT along fast time turns beat
  frequency into range. This gives the range profile of the scene.
- The sequence of chirps is slow time, one chirp per millisecond, which is
  real-world time. A moving target shifts its phase from chirp to chirp; the
  rate of that shift is the Doppler frequency, proportional to radial velocity
  (f_d = 2v / lambda, lambda = c / 5.8 GHz, about 5.2 cm). A short-time FFT
  along slow time gives velocity as a function of time, which is the
  spectrogram.

Micro-Doppler is the set of small Doppler contributions from individual body
parts on top of the bulk body motion. The torso gives a thick slow band; arms
and legs swinging give rhythmic arcs above and below it. Each activity paints a
distinct shape in the velocity versus time image, and that shape is the
fingerprint the classifier learns. Sitting and standing are near mirror images
of each other in Doppler, which predicts they will be the most confused pair and
gives a physical reason to discuss in the results.

### Filename convention `KPxxAyyRz.dat`
- `K` (leading digit) and `Ayy` both encode the activity, 1 to 6.
- `Pxx` is the subject ID within that dataset.
- `Rz` is the repetition.

Three traps, all verified:
1. Repetition zero-padding differs by dataset. December 2017 and February 2019
   UoG use `R01/R02/R03`; the other five use `R1/R2/R3`. Parse with a regex such
   as `R(\d+)` so both work.
2. Subject IDs are not globally unique. The `Pxx` numbers restart per dataset,
   so `P03` in March 2017 and `P03` in February 2019 UoG are different people,
   and `P08` appears in both NG Homes and February 2019 UoG. Any
   subject-independent split must key on the pair (dataset, subject), never the
   bare `Pxx`, or the same person leaks across train and test.
3. The README datasheet maps A04 to pick up and A05 to drink water, while the
   provided `Label_extract4.m` comment maps 4 to drink water and 5 to pick. They
   disagree on positions 4 and 5. Follow the README. This only affects how the
   classes are named in the confusion-matrix discussion, not the training. Note
   the inconsistency in the write-up as something noticed.

### Per-dataset breakdown
| # | Folder | Files | Activities | Subjects | Notes |
|---|--------|------|-----------|----------|-------|
| 1 | 1 December 2017 | 360 | 6 incl. fall | P36 to P56, 20, young males | lab, R0x padding |
| 2 | 2 March 2017 | 48 | 6 | P03, P10 to P12, 4 | 2 reps only |
| 3 | 3 June 2017 | 162 | 6 | P14, P28 to P35, 9 | mixed gender |
| 4 | 4 July 2018 | 288 | 6 | P57 to P72, 16 | common room |
| 5 | 5 February 2019 UoG | 306 | 6 | P01 to P17, 17 | lab, R0x padding |
| 6 | 6 February 2019 NG Homes | 301 | 5, no fall | P08, P18 to P36, 20 | elderly residents |
| 7 | 7 March 2019 West Cumbria | 289 | 6 | P37 to P56, 20 | Age UK centre, elderly |

Total 1754 files. Datasets 1 to 4 are mostly young adults in lab or office
settings; datasets 6 and 7 are older people in care settings, ages up to 98. A
model trained on the young lab data and tested on the elderly care data is an
honest generalization test and a strong result to report.

### Dataset staging
- Stage A, prove it works: dataset 1 (December 2017) only. Clean, all six
  classes, 20 subjects, single site.
- Stage B, generalize: add the others through a config list, decided from
  results not upfront. Dataset 6 has no fall class, handle that explicitly
  (drop fall from that comparison, or treat it as a five-class subset).

The pipeline must be parameterized so switching from one dataset to several is a
config change, not a rewrite.

---

## 6. Verified DSP pipeline

This is a Python port of the provided `DataProcessingExample.m`. It was verified
on `1 December 2017/1P36A01R01.dat` (walking): the output spectrogram shows the
expected walking micro-Doppler, periodic limb arcs on a slower torso band within
about plus or minus 4 m/s. Reference image saved at `sandbox/dsp_check.png`,
script at `sandbox/dsp_check.py`.

Constants from the header: fc 5.8e9, Tsweep 1e-3 s, NTS 128, Bw 400e6,
PRF 1000 Hz.

Steps, with the exact parameters that matched MATLAB:
1. Read the file, split header from data.
2. Reshape the data to (NTS, nc) in Fortran (column-major) order, where
   nc = len(data) // NTS, about 10000.
3. Range FFT along fast time: fftshift(fft(Data_time, axis=0)), then keep the
   upper half rows NTS//2 to NTS, giving 64 range bins.
4. MTI clutter removal: 4th-order Butterworth high-pass with normalized cutoff
   0.0075, applied along slow time per range bin with lfilter (causal, matches
   MATLAB filter). Use ns = oddnumber(nc) - 1 samples.
5. Drop the first range bin of both range and range-MTI arrays, leaving 63 bins.
6. Spectrogram over range bins 10 to 30 (MATLAB 1-indexed), which is Python
   indices 9 to 29 inclusive. Per bin: scipy.signal.spectrogram with
   fs = PRF, window = hamming(200), noverlap = 190, nfft = 800, detrend = False,
   return_onesided = False, mode = 'complex'. Then fftshift along frequency and
   sum the magnitude across bins. Flipud at the end to match MATLAB orientation.
7. Convert Doppler bins to velocity with v = doppler * c / (2 * fc), c = 3e8.
   Full span is about plus or minus 12.9 m/s; the human signature sits within
   plus or minus 6 m/s, so crop to that for display and as a candidate model
   input.

Output spectrogram for a 10 second file is about (800, 981), Doppler bins by
time frames. Display in dB with a 40 dB dynamic range.

Verified code:

```python
import numpy as np
from scipy.signal import butter, lfilter, get_window, spectrogram

def read_dat(path):
    """Read one Glasgow radar .dat file.

    Returns (fc, Tsweep_s, NTS, Bw, data) where data is the complex beat-note
    stream. The first four lines are the header [fc, Tsweep_ms, NTS, Bw]; the
    rest are complex samples written as a+bi.
    """
    with open(path) as f:
        raw = f.read().replace("i", "j")
    v = raw.split()
    fc, tsweep_ms, nts, bw = float(v[0]), float(v[1]), int(float(v[2])), float(v[3])
    data = np.asarray(v[4:], dtype=np.complex128)
    return fc, tsweep_ms / 1000.0, nts, bw, data

def odd_minus_one(n):
    """MATLAB oddnumber(n) - 1: nearest odd value, then minus one (even)."""
    y = int(np.floor(n))
    if y % 2 == 0:
        y = int(np.ceil(n))
    if y % 2 == 0:
        y += 1
    return y - 1

def spectrogram_from_dat(path, bin_lo=9, bin_hi=29):
    """Full chain: .dat -> micro-Doppler spectrogram, plus the velocity axis.

    Ports DataProcessingExample.m. bin_lo and bin_hi are Python indices into the
    range-MTI array (the MATLAB 10..30 window). Returns (spec, velocity) where
    spec is summed |STFT| over the range bins, in linear magnitude.
    """
    fc, Tsweep, NTS, Bw, data = read_dat(path)
    nc = len(data) // NTS
    Data_time = data.reshape(NTS, nc, order="F")
    tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    Data_range = tmp[NTS // 2:NTS, :]

    ns = odd_minus_one(nc)
    b, a = butter(4, 0.0075, "high")
    mti = np.empty((Data_range.shape[0], ns), dtype=np.complex128)
    for k in range(Data_range.shape[0]):
        mti[k, :] = lfilter(b, a, Data_range[k, :ns])
    mti = mti[1:, :]

    PRF = 1.0 / Tsweep
    win = get_window("hamming", 200)
    spec = 0.0
    for rbin in range(bin_lo, bin_hi + 1):
        f, t, S = spectrogram(mti[rbin, :], fs=PRF, window=win, noverlap=190,
                              nfft=800, detrend=False, return_onesided=False,
                              mode="complex", scaling="spectrum")
        spec = spec + np.abs(np.fft.fftshift(S, axes=0))
    spec = np.flipud(spec)
    velocity = np.fft.fftshift(f) * 3e8 / 2 / fc
    return spec, velocity
```

Performance: about 1.5 s per file single threaded, dominated by the text parse.
1754 files take roughly 45 minutes on one core, so parallelize across cores or
cache the spectrograms to .npy once. Caching is required so the notebooks load
precomputed spectrograms quickly.

Open processing choices to settle empirically before the clean build:
- Model input: full spectrogram versus velocity-cropped (plus or minus 6 m/s),
  and the dB normalization scheme (per-image min/max versus a fixed range).
  Decide once and apply consistently to both the CNN input and any image
  features.
- The range-bin window 10 to 30 is fixed from the example. The person may sit
  outside it in some files or datasets. Check before scaling beyond dataset 1.

---

## 7. Decisions made so far

Each entry: what was observed, what was decided, why.

- D1, spectrograms are the working representation. The provided MATLAB and the
  course both center on micro-Doppler spectrograms, and the verified chain
  produces a clean, class-distinct walking signature. Classify from
  spectrograms.
- D2, the Python DSP port matches MATLAB. Verified on a walking file. Use the
  chain in Section 6 as fixed preprocessing so the whole project stays in
  Python and notebooks.
- D3, subject-independent evaluation keyed on (dataset, subject). Subject IDs
  repeat across datasets and refer to different people. Hold out whole people.
  A random split leaks a person across train and test and inflates accuracy.
  This is the central point for the critical-attitude axis.
- D4, follow the README label mapping (A04 pick up, A05 drink), and flag the
  contradiction with the MATLAB comment in the write-up.
- D5, stage the datasets. Build and validate on December 2017 first, then scale
  through a config list. Prove correctness cheaply, and turn the site and age
  variation into an honest generalization test rather than an assumption.

Open questions to resolve with evidence before the clean build:
- Final model input representation and normalization.
- Whether the fixed range-bin window holds across datasets.
- Classical features with a classical model, versus a fine-tuned CNN, versus
  both. Decide from measured accuracy and training cost, not upfront.
- Real per-class counts and any class imbalance once labels are parsed.

---

## 8. Roadmap

Provisional. Each step ends in a finding that justifies the next.

- Phase 0, foundations. Done. Read the README, MATLAB example, label extractor,
  and slide deck. Confirmed the file format and the physical meaning. Verified
  the DSP port against MATLAB. Caught the filename, subject-ID, and label traps.
- Phase 1, dataset index and labels. Parse every filename into (dataset,
  subject, activity, repetition, path). Report per-class and per-dataset counts,
  check imbalance. Implement (dataset, subject) grouping for splitting.
- Phase 2, batch preprocessing and caching. Run the verified DSP over the chosen
  dataset list, cache spectrograms to .npy with a metadata table, one config
  switch for which datasets. Sanity check by plotting one spectrogram per
  activity and confirming the classes look distinct and the range-bin window is
  valid.
- Phase 3, first classifier on December 2017. Decide input representation from
  the plots. Train a simple baseline with subject-independent cross-validation.
  Record accuracy and a confusion matrix. This is the reference point.
- Phase 4, improve and compare. Add a second approach chosen from the Phase 3
  finding (classical features with a classical model, or a fine-tuned CNN on
  MPS, or both). Compare honestly. Error analysis from the confusion matrix tied
  to physics.
- Phase 5, generalization. Scale to more datasets. Test train-on-young-lab,
  test-on-elderly-care. Report what holds and what breaks. This is the headline
  result.
- Phase 6, deliverables. Build the clean notebooks in `notebooks/` from the
  settled sandbox code. Write the one pager and the nine-minute presentation
  outline, every claim traceable to the decisions log, in the writing style of
  Section 3.

---

## 9. First actions for a fresh session

1. Recreate the environment (Section 4) and confirm MPS is available.
2. Re-verify the DSP by running a script that calls `spectrogram_from_dat` on
   `data/Dataset_848/1 December 2017 Dataset/1P36A01R01.dat` and saves the
   image. Confirm it looks like the walking signature before trusting it.
3. Start Phase 1: build the filename index and the per-class, per-subject
   counts. Look at the counts, record the finding, then proceed to Phase 2.

Do not jump ahead to building notebooks. Each notebook follows a sandbox
finding.
