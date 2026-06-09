"""
Phase 2 sanity checks before the full batch:
  1. one micro-Doppler image per activity (dataset 1) -> are classes distinct?
  2. range-bin energy profile across many files -> does the fixed 10..30 window
     actually contain the human return?
Outputs: sandbox/per_activity_ds1.png, sandbox/range_window_ds1.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import radar_pipeline as rp
from scipy.signal import butter, lfilter

rows = rp.build_index([1])
print(f"dataset 1 files: {len(rows)}")

# ---- 1. one image per activity, same subject/rep for a fair comparison ----
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
vel = rp.cropped_velocity_axis()
for act in range(1, 7):
    cand = [r for r in rows if r["activity"] == act and r["subject"] == 36 and r["rep"] == 1]
    r = cand[0]
    spec, velocity, t = rp.compute_spectrogram(r["path"])
    img = rp.db_norm(rp.to_image(spec, velocity))
    ax = axes[(act - 1) // 3, (act - 1) % 3]
    ax.imshow(img, aspect="auto", origin="lower", cmap="jet",
              extent=[0, t[-1], -rp.VEL_CROP, rp.VEL_CROP])
    ax.set_title(f"{act} {rp.ACTIVITY_LONG[act]}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
fig.suptitle("Dataset 1 (Dec 2017), subject P36 rep 1 - one image per activity")
fig.tight_layout()
fig.savefig("sandbox/per_activity_ds1.png", dpi=110)
print("saved sandbox/per_activity_ds1.png")

# ---- 2. range-bin energy profile (validate the 10..30 window) ----
def range_energy(path):
    fc, Tsweep, nts, Bw, data = rp.read_dat(path)
    nc = len(data) // nts
    Data_time = data.reshape(nts, nc, order="F")
    tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    Data_range = tmp[nts // 2:nts, :]
    ns = rp._odd_minus_one(nc)
    b, a = butter(4, rp.MTI_CUTOFF, "high")
    mti = np.empty((Data_range.shape[0], ns), dtype=np.complex128)
    for k in range(Data_range.shape[0]):
        mti[k, :] = lfilter(b, a, Data_range[k, :ns])
    mti = mti[1:, :]                      # 63 bins, matches compute_spectrogram
    return np.abs(mti).sum(axis=1)        # energy per range bin

rng = np.random.default_rng(0)
sample = rng.choice(len(rows), size=min(40, len(rows)), replace=False)
profiles = np.stack([range_energy(rows[i]["path"]) for i in sample])
profiles = profiles / profiles.max(axis=1, keepdims=True)
mean_prof = profiles.mean(axis=0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(len(mean_prof)), mean_prof, lw=2)
ax.axvspan(rp.RANGE_BIN_LO, rp.RANGE_BIN_HI, alpha=0.2, color="green",
           label=f"used bins {rp.RANGE_BIN_LO}..{rp.RANGE_BIN_HI}")
ax.set(xlabel="range bin (after MTI, 0-indexed)", ylabel="mean normalized energy",
       title="Range-bin energy across 40 random dataset-1 files")
ax.legend()
fig.tight_layout()
fig.savefig("sandbox/range_window_ds1.png", dpi=110)
peak = int(np.argmax(mean_prof))
frac_in = mean_prof[rp.RANGE_BIN_LO:rp.RANGE_BIN_HI + 1].sum() / mean_prof.sum()
print(f"saved sandbox/range_window_ds1.png")
print(f"peak energy at bin {peak}; fraction of energy in bins "
      f"{rp.RANGE_BIN_LO}..{rp.RANGE_BIN_HI} = {frac_in:.2f}")
print("per-bin mean energy:", np.round(mean_prof, 2))
