"""
Radar activity classification: shared pipeline.

One verified implementation imported by every notebook, so the DSP parameters,
the file parsing, and the subject-independent splitting are defined once and
cannot drift between notebooks. The signal-processing chain is a direct port of
the provided DataProcessingExample.m and is documented step by step in
notebook 02.

Contents:
  - file IO and filename parsing
  - the FMCW DSP chain (.dat -> micro-Doppler spectrogram)
  - dataset index and (dataset, subject)-keyed grouping
  - spectrogram caching (preprocess once, load fast)
  - model inputs: dB image for the CNN, physical features for the classical model
  - a small CNN (transfer-learned ResNet18) with an MPS train/eval loop
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, lfilter, get_window, spectrogram

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "Dataset_848"
CACHE_ROOT = PROJECT_ROOT / "cache"

# README datasheet numbering (authoritative). The provided Label_extract4.m
# comment swaps 4 and 5; we follow the README (decisions D4).
ACTIVITY_NAMES = {
    1: "walk", 2: "sit", 3: "stand", 4: "pick", 5: "drink", 6: "fall",
}
ACTIVITY_LONG = {
    1: "walking", 2: "sitting down", 3: "standing up",
    4: "picking up", 5: "drinking", 6: "falling",
}

DATASET_NAMES = {
    1: "1 December 2017 Dataset",
    2: "2 March 2017 Dataset",
    3: "3 June 2017 Dataset",
    4: "4 July 2018 Dataset",
    5: "5 February 2019 UoG Dataset",
    6: "6 February 2019 NG Homes Dataset",
    7: "7 March 2019 West Cumbria Dataset",
}

# DSP parameters, fixed from the header and the MATLAB example.
NTS = 128                 # ADC samples per sweep (fast time)
RANGE_BIN_LO = 9          # python index of MATLAB range bin 10
RANGE_BIN_HI = 29         # python index of MATLAB range bin 30 (inclusive)
MTI_CUTOFF = 0.0075       # 4th-order Butterworth high-pass normalized cutoff
STFT_WIN = 200            # slow-time window length
STFT_OVERLAP = 190        # 95% overlap
STFT_NFFT = 800           # 4x zero-pad
C = 3e8

# Model-input geometry: crop to the velocity band that holds the human signature
# and resample to a fixed image so duration differences are normalized.
VEL_CROP = 6.0            # m/s, |v| <= VEL_CROP kept (D: signature lives in +-6)
IMG_H = 128               # velocity bins after resampling
IMG_W = 128               # time bins after resampling

_FILENAME_RE = re.compile(r"^(\d+)P(\d+)A(\d+)R(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# File IO and parsing
# --------------------------------------------------------------------------- #
def read_dat(path):
    """Read one Glasgow radar .dat file.

    The first four whitespace-separated values are the header
    [fc, Tsweep_ms, NTS, Bw]; the rest are complex beat-note samples written as
    a+bi text. Returns (fc, Tsweep_s, NTS, Bw, data) with data complex128.
    """
    with open(path) as f:
        raw = f.read().replace("i", "j")
    v = raw.split()
    fc, tsweep_ms, nts, bw = float(v[0]), float(v[1]), int(float(v[2])), float(v[3])
    data = np.asarray(v[4:], dtype=np.complex128)
    return fc, tsweep_ms / 1000.0, nts, bw, data


def _odd_minus_one(n):
    """MATLAB oddnumber(n) - 1: nearest odd value, then minus one (even)."""
    y = int(np.floor(n))
    if y % 2 == 0:
        y = int(np.ceil(n))
    if y % 2 == 0:
        y += 1
    return y - 1


def parse_filename(path):
    """Filename KPxxAyyRz.dat -> dict(dataset, subject, activity, rep, ...).

    The activity is taken from the Ayy field (the field Label_extract4.m reads);
    the leading digit K is a redundant copy with 3 known typos (decision D6).
    The dataset id is the leading integer of the parent folder name.
    """
    p = Path(path)
    m = _FILENAME_RE.match(p.stem)
    if not m:
        raise ValueError(f"unparseable filename: {p.name}")
    dataset = int(p.parent.name.split()[0])
    subject = int(m.group(2))
    activity = int(m.group(3))          # Ayy field, authoritative
    rep = int(m.group(4))
    return dict(
        dataset=dataset,
        subject=subject,
        activity=activity,
        rep=rep,
        group=f"d{dataset}_s{subject:02d}",   # (dataset, subject) split key
        filename=p.name,
        path=str(p),
    )


# --------------------------------------------------------------------------- #
# DSP chain: .dat -> micro-Doppler spectrogram
# --------------------------------------------------------------------------- #
def compute_spectrogram(path, bin_lo=RANGE_BIN_LO, bin_hi=RANGE_BIN_HI):
    """Port of DataProcessingExample.m: one .dat -> micro-Doppler spectrogram.

    Steps: range FFT along fast time, MTI high-pass along slow time per range
    bin, then a slow-time STFT summed over range bins bin_lo..bin_hi. Returns
    (spec, velocity, t) where spec is the summed |STFT| in linear magnitude with
    rows aligned to ascending velocity (m/s) and columns to time (s).
    """
    fc, Tsweep, nts, Bw, data = read_dat(path)
    nc = len(data) // nts
    Data_time = data.reshape(nts, nc, order="F")

    # Range FFT (fast time), keep the upper half -> 64 range bins.
    tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    Data_range = tmp[nts // 2:nts, :]

    # MTI clutter removal: 4th-order Butterworth high-pass along slow time.
    ns = _odd_minus_one(nc)
    b, a = butter(4, MTI_CUTOFF, "high")
    mti = np.empty((Data_range.shape[0], ns), dtype=np.complex128)
    for k in range(Data_range.shape[0]):
        mti[k, :] = lfilter(b, a, Data_range[k, :ns])
    mti = mti[1:, :]                      # drop first range bin (matches MATLAB)

    # Slow-time STFT per range bin, summed over the selected bins.
    PRF = 1.0 / Tsweep
    win = get_window("hamming", STFT_WIN)
    spec = 0.0
    for rbin in range(bin_lo, bin_hi + 1):
        f, t, S = spectrogram(
            mti[rbin, :], fs=PRF, window=win, noverlap=STFT_OVERLAP,
            nfft=STFT_NFFT, detrend=False, return_onesided=False,
            mode="complex", scaling="spectrum",
        )
        spec = spec + np.abs(np.fft.fftshift(S, axes=0))

    velocity = np.fft.fftshift(f) * C / (2 * fc)   # ascending m/s
    return spec, velocity, t


# --------------------------------------------------------------------------- #
# Model input: crop, resample, dB-normalize
# --------------------------------------------------------------------------- #
def to_image(spec, velocity, vel_crop=VEL_CROP, h=IMG_H, w=IMG_W):
    """Crop the spectrogram to |v|<=vel_crop and resample to (h, w).

    Returns linear-magnitude float32. Cropping trims empty Doppler bands; the
    fixed velocity window means row index maps linearly to velocity for every
    file, and resampling time to a fixed width normalizes small duration
    differences. dB and per-image scaling are applied later (db_norm).
    """
    mask = np.abs(velocity) <= vel_crop
    cropped = spec[mask, :]
    img = Image.fromarray(cropped.astype(np.float32), mode="F")
    img = img.resize((w, h), Image.BILINEAR)   # PIL size is (width, height)
    return np.asarray(img, dtype=np.float32)


def db_norm(img, dyn_range=40.0):
    """Linear magnitude -> dB, clipped to a per-image dyn_range, scaled to [0,1].

    Per-image max as the 0 dB reference matches the MATLAB display convention and
    removes the absolute-gain differences between recordings.
    """
    db = 20.0 * np.log10(img + 1e-12)
    db = db - db.max()
    db = np.clip(db, -dyn_range, 0.0)
    return (db + dyn_range) / dyn_range


def cropped_velocity_axis(vel_crop=VEL_CROP, h=IMG_H):
    """Velocity (m/s) for each row of a to_image() output (ascending)."""
    return np.linspace(-vel_crop, vel_crop, h)


# --------------------------------------------------------------------------- #
# Dataset index
# --------------------------------------------------------------------------- #
def build_index(dataset_ids):
    """Scan the given datasets and return a list of parsed-file dicts, sorted."""
    rows = []
    for did in dataset_ids:
        folder = DATA_ROOT / DATASET_NAMES[did]
        for f in sorted(folder.glob("*.dat")):
            rows.append(parse_filename(f))
    rows.sort(key=lambda r: (r["dataset"], r["subject"], r["activity"], r["rep"]))
    return rows


# --------------------------------------------------------------------------- #
# Caching: preprocess once to (h, w) linear-magnitude .npy + a single index .npz
# --------------------------------------------------------------------------- #
def _cache_key(row):
    return f"d{row['dataset']}_{Path(row['path']).stem}"


def preprocess_dataset(dataset_ids, out_dir=CACHE_ROOT, force=False, n_jobs=None,
                       progress=True):
    """Compute and cache the (IMG_H, IMG_W) linear spectrogram for each file.

    Writes one .npy per file plus index.npz (arrays: keys, dataset, subject,
    activity, rep, group). Idempotent: existing .npy are skipped unless force.
    Parallel across files with a process pool. Returns the index as a dict of
    arrays. Run once; the modeling notebooks call load_cache().
    """
    import os

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_index(dataset_ids)
    todo = [r for r in rows
            if force or not (out_dir / f"{_cache_key(r)}.npy").exists()]

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if todo:
        args = [(r["path"], str(out_dir / f"{_cache_key(r)}.npy")) for r in todo]
        it = _imap(args, n_jobs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(args), desc="preprocess")
        for _ in it:
            pass

    keys = np.array([_cache_key(r) for r in rows])
    idx = dict(
        keys=keys,
        dataset=np.array([r["dataset"] for r in rows]),
        subject=np.array([r["subject"] for r in rows]),
        activity=np.array([r["activity"] for r in rows]),
        rep=np.array([r["rep"] for r in rows]),
        group=np.array([r["group"] for r in rows]),
        filename=np.array([r["filename"] for r in rows]),
    )
    np.savez(out_dir / "index.npz", **idx)
    return idx


def _process_one(arg):
    """Worker: compute the cropped/resampled image for one file and save it."""
    path, out_path = arg
    if Path(out_path).exists():
        return out_path
    spec, velocity, _ = compute_spectrogram(path)
    img = to_image(spec, velocity)
    np.save(out_path, img)
    return out_path


def _imap(args, n_jobs):
    from concurrent.futures import ProcessPoolExecutor
    if n_jobs == 1:
        for a in args:
            yield _process_one(a)
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        yield from ex.map(_process_one, args)


def load_cache(dataset_ids=None, out_dir=CACHE_ROOT):
    """Load cached images and metadata.

    Returns (X, meta) where X is (N, IMG_H, IMG_W) linear magnitude float32 and
    meta is a dict of aligned metadata arrays. If dataset_ids is given, only
    those datasets are returned.
    """
    out_dir = Path(out_dir)
    idx = np.load(out_dir / "index.npz", allow_pickle=True)
    meta = {k: idx[k] for k in idx.files}
    if dataset_ids is not None:
        sel = np.isin(meta["dataset"], list(dataset_ids))
        meta = {k: v[sel] for k, v in meta.items()}
    X = np.stack([np.load(out_dir / f"{k}.npy") for k in meta["keys"]])
    return X.astype(np.float32), meta


# --------------------------------------------------------------------------- #
# Classical features: physical micro-Doppler descriptors
# --------------------------------------------------------------------------- #
def extract_features(img_linear, velocity=None):
    """Physical micro-Doppler features from one linear-magnitude spectrogram.

    Treats each time column as an energy distribution over velocity and
    summarizes the Doppler centroid and bandwidth trajectories plus the
    torso/limb energy split and the temporal envelope. These are the descriptors
    the micro-Doppler literature uses, chosen so a classical model stays
    interpretable. Returns a 1-D feature vector.
    """
    if velocity is None:
        velocity = cropped_velocity_axis(h=img_linear.shape[0])
    p = img_linear.astype(np.float64)
    col_energy = p.sum(axis=0) + 1e-12             # energy per time frame
    pn = p / col_energy                             # per-frame distribution

    centroid = (velocity[:, None] * pn).sum(axis=0)             # mean velocity(t)
    var = ((velocity[:, None] - centroid[None, :]) ** 2 * pn).sum(axis=0)
    bandwidth = np.sqrt(np.maximum(var, 0))                     # spread(t)

    # Torso (|v|<1) vs limb (|v|>=1) energy fraction over the whole image.
    torso = np.abs(velocity) < 1.0
    e_total = p.sum() + 1e-12
    torso_frac = p[torso, :].sum() / e_total
    limb_frac = 1.0 - torso_frac

    # Positive vs negative Doppler energy (approach vs recede asymmetry).
    pos = velocity > 0
    neg = velocity < 0
    pos_frac = p[pos, :].sum() / e_total
    neg_frac = p[neg, :].sum() / e_total

    env = col_energy / col_energy.max()             # normalized energy envelope

    def stats(x):
        return [x.mean(), x.std(), np.percentile(x, 10), np.percentile(x, 90),
                x.min(), x.max()]

    feats = []
    feats += stats(centroid)
    feats += stats(bandwidth)
    feats += stats(env)
    feats += [torso_frac, limb_frac, pos_frac, neg_frac]
    feats += [np.abs(centroid).max(),                # peak |radial velocity|
              bandwidth.max(),                        # peak Doppler spread
              centroid.max() - centroid.min(),        # centroid swing
              float((env > 0.5).mean())]              # active-time fraction
    return np.asarray(feats, dtype=np.float32)


FEATURE_NAMES = (
    [f"centroid_{s}" for s in ("mean", "std", "p10", "p90", "min", "max")] +
    [f"bandwidth_{s}" for s in ("mean", "std", "p10", "p90", "min", "max")] +
    [f"envelope_{s}" for s in ("mean", "std", "p10", "p90", "min", "max")] +
    ["torso_frac", "limb_frac", "pos_frac", "neg_frac",
     "peak_abs_centroid", "peak_bandwidth", "centroid_swing", "active_frac"]
)


def feature_matrix(X_linear):
    """extract_features over a stack -> (N, n_features)."""
    return np.stack([extract_features(x) for x in X_linear])


# --------------------------------------------------------------------------- #
# CNN: transfer-learned ResNet18 on dB images (MPS)
# --------------------------------------------------------------------------- #
def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_cnn(n_classes, pretrained=True):
    """ResNet18 adapted to 1-channel dB spectrograms with an n_classes head.

    The first conv is reduced to one input channel (the pretrained RGB weights
    are averaged) so the micro-Doppler image is fed directly without faking 3
    channels. Small enough to fine-tune on MPS in minutes.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    net = resnet18(weights=weights)
    w = net.conv1.weight.data
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        net.conv1.weight.data = w.mean(dim=1, keepdim=True)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


def db_stack(X_linear, dyn_range=40.0):
    """db_norm over a stack -> (N, IMG_H, IMG_W) in [0,1]."""
    return np.stack([db_norm(x, dyn_range) for x in X_linear]).astype(np.float32)


def train_cnn(net, Xtr, ytr, Xva, yva, epochs=20, lr=1e-3, batch=32,
              weight=None, seed=0, verbose=False):
    """Fine-tune the CNN. X are dB images in [0,1]; returns the best-val state.

    Tracks validation accuracy each epoch and keeps the best weights. weight is
    an optional per-class loss weight (numpy) for class imbalance.
    """
    import torch
    import torch.nn as nn
    from copy import deepcopy

    torch.manual_seed(seed)
    dev = get_device()
    net = net.to(dev)
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)
    ytr_t = torch.from_numpy(ytr).long()
    Xva_t = torch.from_numpy(Xva).unsqueeze(1).to(dev)

    w_t = None if weight is None else torch.tensor(weight, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w_t)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n = len(Xtr_t)
    g = torch.Generator().manual_seed(seed)
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr_t[idx].to(dev)
            yb = ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
        acc = _eval_acc(net, Xva_t, yva, dev)
        if acc > best_acc:
            best_acc, best_state = acc, deepcopy(net.state_dict())
        if verbose:
            print(f"  epoch {ep + 1:2d}/{epochs}  val_acc={acc:.3f}")
    net.load_state_dict(best_state)
    return net, best_acc


def _eval_acc(net, Xva_t, yva, dev):
    import torch
    net.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xva_t), 128):
            out = net(Xva_t[i:i + 128])
            preds.append(out.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return float((pred == yva).mean())


def cnn_predict(net, X_db):
    """Class predictions for dB images (N, H, W)."""
    import torch
    dev = get_device()
    net = net.to(dev).eval()
    Xt = torch.from_numpy(X_db).unsqueeze(1).to(dev)
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xt), 128):
            preds.append(net(Xt[i:i + 128]).argmax(1).cpu().numpy())
    return np.concatenate(preds)
