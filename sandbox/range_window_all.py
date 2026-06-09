"""
Re-check the fixed range-bin window 9..29 on every dataset (D10 flagged that
elderly-care standoff range may differ). Prints peak bin and energy fraction in
the window per dataset, using 25 random files each.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import radar_pipeline as rp
from scipy.signal import butter, lfilter

def range_energy(path):
    fc, Tsweep, nts, Bw, data = rp.read_dat(path)
    nc = len(data) // nts
    tmp = np.fft.fftshift(np.fft.fft(data.reshape(nts, nc, order="F"), axis=0), axes=0)
    Data_range = tmp[nts // 2:nts, :]
    ns = rp._odd_minus_one(nc)
    b, a = butter(4, rp.MTI_CUTOFF, "high")
    mti = np.empty((Data_range.shape[0], ns), dtype=np.complex128)
    for k in range(Data_range.shape[0]):
        mti[k, :] = lfilter(b, a, Data_range[k, :ns])
    return np.abs(mti[1:, :]).sum(axis=1)

rng = np.random.default_rng(0)
print(f"window = bins {rp.RANGE_BIN_LO}..{rp.RANGE_BIN_HI}")
for d in range(1, 8):
    rows = rp.build_index([d])
    idx = rng.choice(len(rows), size=min(25, len(rows)), replace=False)
    prof = np.stack([range_energy(rows[i]["path"]) for i in idx])
    prof = (prof / prof.max(axis=1, keepdims=True)).mean(axis=0)
    peak = int(np.argmax(prof))
    frac = prof[rp.RANGE_BIN_LO:rp.RANGE_BIN_HI + 1].sum() / prof.sum()
    print(f"dataset {d}: peak bin {peak:2d}, energy in window = {frac:.2f}")
