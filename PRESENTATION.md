# Presentation outline (~9 minutes + Q&A)

Nine slides, roughly one minute each. Speaker notes are what to say, not what to
put on the slide. Keep slides to a title, one figure, and a few words.

---

### 1. Title and the question (40 s)
Slide: project title, course, three names. One line: "classify six daily
activities from radar micro-Doppler, and find out how far such a model travels
to a new population."
Notes: frame it as the open problem the professor set: a reasonable, justified
pipeline, with the honest generalization test as the interesting part.

### 2. The data and three traps (60 s)
Slide: the per-activity / per-dataset count table from notebook 01.
Notes: Glasgow INSHEP, 1754 files, FMCW 5.8 GHz / 400 MHz / 1 ms / 128 samples,
seven campaigns, young labs vs elderly care. Three traps we handled: activity is
read from the `Ayy` field (three leading-digit typos); subject IDs collide across
campaigns, so we split on `(dataset, subject)`; fall is scarce and missing from
the care sites. Mention the README/MATLAB label disagreement on 4 and 5.

### 3. From raw radar to a picture (75 s)
Slide: the DSP chain as range-time then micro-Doppler (notebook 02 figures).
Notes: one complex sample is the de-chirped beat note. FFT along fast time gives
range; MTI high-pass along slow time removes static clutter; STFT along slow time
gives velocity over time. This is the course chain (FMCW, beat note, fast/slow
time, range FFT, MTI, STFT). We ported the provided MATLAB and verified it on a
walking file.

### 4. What each activity looks like (45 s)
Slide: the six-panel micro-Doppler grid (notebook 02).
Notes: walking is sustained and periodic; the rest are transients; falling is the
strongest, broadest burst. This is the fingerprint the classifier reads. Input is
cropped to +-6 m/s, fixed 128x128, per-image dB.

### 5. How we evaluate, and why it matters (45 s)
Slide: a cartoon of subject-independent splitting (hold out whole people).
Notes: the central methodological point. A random split leaks the same person
into train and test and inflates accuracy. We hold out whole `(dataset, subject)`
groups. Every number in the talk is subject-independent.

### 6. Two models, one comparison (60 s)
Slide: the comparison table (features+SVM vs ResNet18) from notebook 04.
Notes: an interpretable model (26 physical micro-Doppler descriptors into an SVM)
against a fine-tuned ResNet18. On dataset 1 both reach ~0.96. The SVM matches the
CNN at a fraction of the cost and stays interpretable. On clean data, capacity is
not the bottleneck.

### 7. Where the errors are, and why (50 s)
Slide: the dataset-1 confusion matrix (notebook 03).
Notes: walking near perfect. Dominant error pick vs drink, both low-velocity arm
motions with similar Doppler. The sit/stand mirror pair, which we expected to be
the hard one, is resolved by the positive/negative-Doppler features. Conclusions
tied to physics.

### 8. The headline: lab to elderly care (90 s)
Slide: bar chart, within-source CV vs the two care datasets (notebook 05), plus
the target confusion matrices.
Notes: train on the lab pool, test on the care sites. Accuracy drops from ~0.89
to ~0.77, and to ~0.74 for West Cumbria. The CNN does not generalize better; it
is comparable and unstable across seeds. Two measured causes: elderly pick/drink
motions are slower and smaller, so that confusion deepens; and the fixed range
window captures only ~0.4-0.5 of the care-home return because those subjects
stand closer to the radar (the range-energy figure from notebook 02).

### 9. Conclusions and next steps (40 s)
Slide: three bullets.
Notes: micro-Doppler separates these activities; simple features match a CNN on
clean data; the real story is the generalization gap, which is about population
and geometry, not model size. Next: re-centre the range window per site, a little
in-domain care data for domain adaptation, and report fall separately.

---

## Likely questions and short answers
- Why not deep-learn end to end from the raw signal? The spectrogram is the
  physically meaningful representation the course teaches, and on this data
  volume handcrafted features already match a CNN; a heavier model is not
  justified.
- Why does the SVM tie the CNN? The classes are well separated in a few physical
  dimensions (centroid, bandwidth, direction, envelope); the CNN has nothing
  extra to exploit on clean single-site data.
- Is the generalization drop just overfitting? No: it is subject-independent
  within source too (~0.89). The drop is domain shift (age, site, standoff
  geometry), which we measured.
- Why drop fall for the cross-site test? The care sites barely recorded it (ds7
  has none), so it cannot be scored there; we use the five shared classes and
  report fall separately.
- What would most improve cross-site accuracy? Re-centring the range window per
  site (it currently clips the care return) and adding a small amount of
  in-domain data.
