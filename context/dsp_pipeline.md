# DSP pipeline (verified)

Python port of the provided `DataProcessingExample.m`. Verified on
`1 December 2017/1P36A01R01.dat` (walking): the output spectrogram shows the
expected walking micro-Doppler — periodic limb-swing peaks on a slower torso
Doppler, contained within roughly ±4 m/s. Reference image:
`sandbox/dsp_check.png`. Implementation: `sandbox/dsp_check.py`.

## Steps and exact parameters
Constants from the header: `fc = 5.8e9`, `Tsweep = 1e-3 s`, `NTS = 128`,
`Bw = 400e6`, `PRF = 1/Tsweep = 1000 Hz`.

1. **Read** the `.dat`, split header from data (see `data.md`).
2. **Reshape** the data to `(NTS, nc)` using Fortran/column-major order
   (`reshape(NTS, nc, order="F")`), where `nc = len(data)//NTS` (≈10000).
3. **Range FFT** along fast time: `tmp = fftshift(fft(Data_time, axis=0))`,
   then keep the upper half `Data_range = tmp[NTS//2:NTS, :]` → 64 range bins.
4. **MTI clutter removal**: 4th-order Butterworth high-pass, normalized cutoff
   `0.0075` (`scipy.signal.butter(4, 0.0075, 'high')`), applied along slow time
   per range bin with `lfilter` (causal, matches MATLAB `filter`). Use
   `ns = oddnumber(nc)-1` samples (even length; ≈ nc).
5. **Drop the first range bin** of both `Data_range` and `Data_range_MTI`
   (matches MATLAB `(2:end,:)`), leaving 63 bins.
6. **Spectrogram (slow-time STFT)** over range bins **10..30** (MATLAB
   1-indexed) = Python indices `9..29` inclusive. Per bin:
   `scipy.signal.spectrogram(x, fs=PRF, window=hamming(200), noverlap=190,
   nfft=800, detrend=False, return_onesided=False, mode='complex')`, then
   `fftshift` along frequency and sum `|STFT|` across the bins. `flipud` at the
   end to match the MATLAB orientation.
   - Window length 200, overlap 0.95 (→190), pad factor 4 (→ nfft 800).
7. **Axes**: Doppler bins → velocity via `v = doppler * c / (2*fc)`,
   `c = 3e8`. Full velocity span is ±12.9 m/s; the human signature lives within
   ±6 m/s — crop to ±6 for display and as a candidate model input.

Output spectrogram for a 10 s file: ~`(800, 981)` (Doppler bins × time frames),
shown in dB with a 40 dB dynamic range (`vmax`, `vmax-40`).

## Notes / open choices (decide empirically)
- **Velocity crop** for model input (e.g. ±6 m/s) — trims empty Doppler bands,
  shrinks the image. Not yet fixed.
- **Range-bin selection** is fixed at 10..30 from the example; the person may
  sit outside that band in some files/datasets. Check before scaling.
- **dB normalization** (per-image min/max vs fixed range) affects both the CNN
  input and any image-based features. Decide once, apply consistently.

## Performance
~1.5 s/file single-threaded (text parse dominates). 1754 files ≈ ~45 min on one
core; parallelize across cores or cache spectrograms to `.npy` once. Caching is
required so the professor notebooks load precomputed spectrograms quickly.
