# Human Activity Classification with FMCW Radar Micro-Doppler

EE4775 Object Classification with Radar, TU Delft. Group of three.

## Problem and data
We classify radar recordings of people performing six daily activities (walking,
sitting down, standing up, picking up an object, drinking, falling) from their
micro-Doppler signatures. The data is the University of Glasgow INSHEP set: 1754
`.dat` recordings from an Ancortek FMCW radar (5.8 GHz carrier, 400 MHz
bandwidth, 1 ms sweep, 128 samples per sweep, 1 kHz PRF), collected across seven
campaigns. Datasets 1-5 are mostly young adults in labs and offices; datasets 6
and 7 are elderly people in care settings (ages up to 98).

Parsing the filenames surfaced three facts that shape the work. Activity is read
from the `Ayy` field (the field the provided `Label_extract4.m` uses); the
redundant leading digit disagrees with it in three files (typos). Subject IDs
restart per campaign, so the same `Pxx` is different people in different
datasets; all evaluation therefore holds out whole `(dataset, subject)` pairs.
Fall is the minority class (197 vs ~312 per class) and is essentially absent from
the two care sites, so cross-site tests use the five shared classes. The README
and the MATLAB comment also disagree on the names of activities 4 and 5; we
follow the README and note it.

## Signal processing
Each recording is turned into a velocity-vs-time image, following the provided
`DataProcessingExample.m`: an FFT along fast time (within a sweep) gives range; a
4th-order Butterworth high-pass along slow time (MTI) removes static clutter; a
short-time FFT along slow time, summed over the target range bins, gives the
micro-Doppler spectrogram. We crop to the human velocity band (|v| <= 6 m/s),
resample to a fixed 128x128 image so recording-length differences do not leak in,
and normalize to dB referenced to each image's maximum over a 40 dB range. The
same input feeds both classifiers. Preprocessing is run once and cached.

## Models and evaluation
We compare two approaches on the same subject-independent splits. The first is
interpretable: 26 physical descriptors per spectrogram (Doppler centroid and
bandwidth trajectories, torso/limb and approach/recede energy split, temporal
envelope) into an RBF SVM. The second is a ResNet18 fine-tuned on the dB image
(first convolution reduced to one channel), trained on the Apple-silicon GPU.

On dataset 1 (clean, balanced, single site) both reach about 0.96 accuracy under
5-fold subject-independent cross-validation; the SVM is as accurate as the CNN at
a fraction of the cost and fully interpretable. The errors are physical: walking
is near perfect, the dominant confusion is pick versus drink (both low-velocity
arm motions with similar Doppler), and the sit/stand mirror pair is largely
resolved by the directional Doppler features.

## Generalization, the headline
Training on the lab/university pool (datasets 1-5) and testing on the elderly
care sites (datasets 6-7), five-class, accuracy falls from ~0.89 within the lab
pool to ~0.77 on the care data, and to ~0.74 for West Cumbria. The CNN does not
generalize better than the SVM; the extra capacity buys no cross-domain
robustness. Two measured causes: the elderly perform pick and drink more slowly
and with smaller motions, deepening that confusion; and the range-bin window
fixed on the lab data captures only ~0.4-0.5 of the care-home return because
those subjects stand closer to the radar, a geometry shift on top of the age and
site shift.

## Conclusions
Micro-Doppler spectrograms separate these activities well, and on clean data
simple interpretable features match a fine-tuned CNN. The honest result is the
generalization gap: a model is only as good as the population and geometry it was
trained on. Concrete next steps are to re-centre the range window per site, add a
small amount of in-domain care data (domain adaptation), and report fall
separately given its scarcity at the care sites.
