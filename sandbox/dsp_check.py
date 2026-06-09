"""
Verify a Python port of the provided MATLAB DSP (DataProcessingExample.m)
on a single .dat file. Goal: confirm the chain reproduces a sensible
range-time map and micro-Doppler spectrogram before trusting it on 1754 files.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, get_window, spectrogram

DATA = "../data/Dataset_848/1 December 2017 Dataset/1P36A01R01.dat"  # activity 1 = walking


def read_dat(path):
    with open(path) as f:
        raw = f.read().replace("i", "j")
    v = raw.split()
    fc, tsweep_ms, nts, bw = float(v[0]), float(v[1]), int(float(v[2])), float(v[3])
    data = np.asarray(v[4:], dtype=np.complex128)
    return fc, tsweep_ms, nts, bw, data


def odd_minus_one(n):
    # MATLAB oddnumber(n)-1: nearest odd >= behaviour, then -1 -> even
    y = int(np.floor(n))
    if y % 2 == 0:
        y = int(np.ceil(n))
    if y % 2 == 0:
        y += 1
    return y - 1


fc, tsweep_ms, NTS, Bw, data = read_dat(DATA)
Tsweep = tsweep_ms / 1000.0
fs = NTS / Tsweep
nc = len(data) // NTS
print(f"fc={fc:g} Tsweep={Tsweep:g}s NTS={NTS} Bw={Bw:g} | samples={len(data)} chirps={nc} dur={nc*Tsweep:g}s")

# fast-time FFT -> range, keep upper half (MATLAB tmp(NTS/2+1:NTS,:))
Data_time = data.reshape(NTS, nc, order="F")
tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
Data_range = tmp[NTS // 2:NTS, :]            # 64 range bins

ns = odd_minus_one(nc)
b, a = butter(4, 0.0075, "high")
Data_range_MTI = np.empty((Data_range.shape[0], ns), dtype=np.complex128)
for k in range(Data_range.shape[0]):
    Data_range_MTI[k, :] = lfilter(b, a, Data_range[k, :ns])

Data_range_MTI = Data_range_MTI[1:, :]       # drop first bin -> 63 bins
Data_range = Data_range[1:, :ns]
print(f"range-time shape: {Data_range_MTI.shape} (range bins x sweeps), ns={ns}")

# spectrogram over range bins 10..30 (MATLAB 1-indexed) = python 9..29
PRF = 1.0 / Tsweep
win_len, overlap, nfft = 200, 190, 800
win = get_window("hamming", win_len)
spec = 0.0
for rbin in range(9, 30):
    f, t, S = spectrogram(Data_range_MTI[rbin, :], fs=PRF, window=win,
                          noverlap=overlap, nfft=nfft, detrend=False,
                          return_onesided=False, mode="complex", scaling="spectrum")
    spec = spec + np.abs(np.fft.fftshift(S, axes=0))
spec = np.flipud(spec)
doppler = np.fft.fftshift(f)
velocity = doppler * 3e8 / 2 / fc
print(f"spectrogram shape: {spec.shape} (doppler bins x time frames)")
print(f"velocity axis: {velocity.min():.2f} .. {velocity.max():.2f} m/s")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
rt = 20 * np.log10(np.abs(Data_range_MTI) + 1e-12)
im0 = ax[0].imshow(rt, aspect="auto", origin="lower", cmap="jet",
                   vmax=rt.max(), vmin=rt.max() - 60)
ax[0].set(title="Range-time (after MTI)", xlabel="sweep", ylabel="range bin", ylim=(0, 63))
fig.colorbar(im0, ax=ax[0])

sd = 20 * np.log10(spec + 1e-12)
t_axis = np.linspace(0, nc * Tsweep, spec.shape[1])
im1 = ax[1].pcolormesh(t_axis, velocity, sd, cmap="jet", vmax=sd.max(), vmin=sd.max() - 40)
ax[1].set(title="micro-Doppler (walking)", xlabel="time [s]", ylabel="velocity [m/s]", ylim=(-6, 6))
fig.colorbar(im1, ax=ax[1])
fig.tight_layout()
fig.savefig("dsp_check.png", dpi=110)
print("saved dsp_check.png")
