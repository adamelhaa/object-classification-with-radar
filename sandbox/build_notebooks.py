"""
Generate the five professor-facing notebooks from the settled sandbox code.
Each notebook mirrors a sandbox script that already produced the reported
numbers. Run this, then execute the notebooks with nbconvert.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parent.parent / "notebooks"


def build(name, cells):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python (radar)", "language": "python", "name": "radar"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, OUT / name)
    print("wrote", name)


def md(s):
    return new_markdown_cell(s.strip("\n"))


def code(s):
    return new_code_cell(s.strip("\n"))


# --------------------------------------------------------------------------- #
# 01 - dataset index and labels
# --------------------------------------------------------------------------- #
build("01_dataset_index.ipynb", [
md("""
# 01 - Dataset index and labels

Human activity classification from FMCW radar micro-Doppler (EE4775).
This notebook parses every `.dat` filename into (dataset, subject, activity,
repetition), reports the class and site distribution, and fixes the unit used for
subject-independent splitting. Shared code lives in `radar_pipeline.py`.
"""),
code("""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import radar_pipeline as rp

ALL = [1, 2, 3, 4, 5, 6, 7]
rows = rp.build_index(ALL)
df = pd.DataFrame(rows)
df["activity_name"] = df["activity"].map(rp.ACTIVITY_NAMES)
print(f"{len(df)} files across {df.dataset.nunique()} datasets")
df.head()
"""),
md("""
## Class and site distribution

The activity comes from the `Ayy` field of the filename, which is the field the
provided `Label_extract4.m` reads. The leading digit `K` is a redundant copy.
"""),
code("""
counts = df.activity_name.value_counts().reindex(
    [rp.ACTIVITY_NAMES[i] for i in range(1, 7)])
print(counts)
fig, ax = plt.subplots(figsize=(6, 3))
counts.plot.bar(ax=ax, color="steelblue")
ax.set(ylabel="files", title="Files per activity (all datasets)")
plt.tight_layout(); plt.show()
"""),
code("""
mat = (df.pivot_table(index="activity_name", columns="dataset",
                      values="path", aggfunc="count", fill_value=0)
         .reindex([rp.ACTIVITY_NAMES[i] for i in range(1, 7)]))
fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(mat.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, mat.values[i, j], ha="center", va="center", fontsize=8)
ax.set(xlabel="dataset", title="Activity x dataset file counts")
plt.tight_layout(); plt.show()
mat
"""),
md("""
## Findings

- The five non-fall activities have ~311-312 files each; **fall has only 197**.
  Fall is **absent from dataset 7** (West Cumbria) and nearly absent from
  dataset 6 (NG Homes, subject P08 only). The care sites did not record falls at
  scale, so any train/test pairing that involves them is handled as a 5-class
  problem (notebook 05).
- The README datasheet maps `A04=pick up`, `A05=drink`; the provided
  `Label_extract4.m` comment swaps them. We follow the README and flag it.
"""),
code("""
# Leading-digit K vs Ayy: 3 files disagree (typos in the redundant K field).
typos = [r for r in rows
         if int(rp.Path(r["path"]).stem[0]) != r["activity"]]
print("K vs Ayy mismatches:", [t["filename"] for t in typos])
"""),
md("""
## The splitting unit

Subject IDs restart per dataset, so the same `Pxx` denotes different people in
different datasets. Splitting must hold out whole **(dataset, subject)** pairs,
not bare IDs, or the same person leaks across train and test and inflates
accuracy. The `group` column already encodes this key.
"""),
code("""
n_people = df.group.nunique()
bare_collisions = (df.groupby("subject").dataset.nunique() > 1).sum()
print(f"distinct (dataset, subject) people: {n_people}")
print(f"bare subject IDs appearing in >1 dataset: {bare_collisions}")
print("example group keys:", sorted(df.group.unique())[:5])
"""),
])


# --------------------------------------------------------------------------- #
# 02 - preprocessing (DSP chain + caching)
# --------------------------------------------------------------------------- #
build("02_preprocessing.ipynb", [
md("""
# 02 - From `.dat` to micro-Doppler spectrogram

Port of the provided `DataProcessingExample.m`. One FMCW recording becomes a
velocity-vs-time image: range FFT along fast time, MTI clutter removal along slow
time, then a short-time FFT along slow time. The canonical implementation is
`radar_pipeline.compute_spectrogram`; here we walk it step by step on one walking
file so every step is defensible, then cache all files to disk.
"""),
code("""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, get_window, spectrogram
import radar_pipeline as rp

walk = [r for r in rp.build_index([1])
        if r["activity"] == 1 and r["subject"] == 36 and r["rep"] == 1][0]
fc, Tsweep, NTS, Bw, data = rp.read_dat(walk["path"])
nc = len(data) // NTS
PRF = 1.0 / Tsweep
print(f"fc={fc:g} Hz  Bw={Bw:g} Hz  Tsweep={Tsweep*1e3:g} ms  "
      f"NTS={NTS}  chirps={nc}  duration={nc*Tsweep:g} s  PRF={PRF:g} Hz")
"""),
md("""
## Step 1 - fast time to range

The stream is reshaped to (samples-per-sweep, chirps) in column-major order. An
FFT along fast time turns the beat frequency of each chirp into range; we keep
the upper half (64 range bins).
"""),
code("""
Data_time = data.reshape(NTS, nc, order="F")
tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
Data_range = tmp[NTS // 2:NTS, :]
print("range x slow-time:", Data_range.shape)
"""),
md("""
## Step 2 - MTI clutter removal

Static scene reflections (walls, furniture) sit at zero Doppler. A 4th-order
Butterworth high-pass along slow time, per range bin, removes them and keeps the
moving target. We then drop the first range bin, matching the MATLAB example.
"""),
code("""
ns = rp._odd_minus_one(nc)
b, a = butter(4, rp.MTI_CUTOFF, "high")
mti = np.empty((Data_range.shape[0], ns), dtype=np.complex128)
for k in range(Data_range.shape[0]):
    mti[k, :] = lfilter(b, a, Data_range[k, :ns])
mti = mti[1:, :]

fig, ax = plt.subplots(figsize=(8, 3))
rt = 20 * np.log10(np.abs(mti) + 1e-12)
im = ax.imshow(rt, aspect="auto", origin="lower", cmap="jet",
               vmax=rt.max(), vmin=rt.max() - 60)
ax.set(xlabel="sweep (slow time)", ylabel="range bin",
       title="Range-time after MTI (walking)")
fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
"""),
md("""
The target return concentrates around range bins ~10-20. The spectrogram is
computed over bins 10..30 (MATLAB 1-indexed), summing their contributions.

## Step 3 - slow time to Doppler (STFT)

A short-time FFT along slow time, per selected range bin, gives velocity over
time. Hamming window of 200 sweeps, 95% overlap, 4x zero-padding. The Doppler
axis converts to velocity with `v = f_d * c / (2 * fc)`.
"""),
code("""
# Single range bin vs the summed micro-Doppler (mirrors compute_spectrogram).
win = get_window("hamming", rp.STFT_WIN)
f, t, S = spectrogram(mti[14, :], fs=PRF, window=win, noverlap=rp.STFT_OVERLAP,
                      nfft=rp.STFT_NFFT, detrend=False, return_onesided=False,
                      mode="complex", scaling="spectrum")
single = np.abs(np.fft.fftshift(S, axes=0))
spec, velocity, t = rp.compute_spectrogram(walk["path"])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a_, img, ttl in [(ax[0], single, "single range bin 14"),
                     (ax[1], spec, "summed over bins 10..30")]:
    d = 20 * np.log10(img + 1e-12)
    a_.imshow(d, aspect="auto", origin="lower", cmap="jet",
              extent=[0, t[-1], velocity[0], velocity[-1]],
              vmax=d.max(), vmin=d.max() - 40)
    a_.set(xlabel="time [s]", ylabel="velocity [m/s]", ylim=(-6, 6), title=ttl)
plt.tight_layout(); plt.show()
"""),
md("""
Summing range bins raises the limb arcs out of the noise. This walking signature
(rhythmic limb arcs on a slower torso band, within +-4 m/s) matches the expected
micro-Doppler and confirms the port.

## Model input

Crop to |v| <= 6 m/s (the signature lives there), resample to a fixed 128x128 so
recording-length differences are normalized, and convert to dB referenced to the
per-image maximum over a 40 dB range. The same input feeds both models.
"""),
code("""
img = rp.to_image(spec, velocity)
img_db = rp.db_norm(img)
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img_db, origin="lower", cmap="jet",
          extent=[0, 1, -rp.VEL_CROP, rp.VEL_CROP], aspect="auto")
ax.set(title="model input (128x128, dB)", xlabel="time (normalized)",
       ylabel="velocity [m/s]")
plt.tight_layout(); plt.show()
print("input shape:", img_db.shape, "range:", img_db.min(), img_db.max())
"""),
md("""
## Cache all files

`preprocess_dataset` runs the chain over every file once and caches the 128x128
linear-magnitude image to `.npy` (parallel across cores). It is idempotent, so
re-running is cheap. The modeling notebooks load the cache.
"""),
code("""
idx = rp.preprocess_dataset([1, 2, 3, 4, 5, 6, 7], progress=False)
X, meta = rp.load_cache([1])
print("cached files total:", len(idx["keys"]))
print("dataset 1 loaded:", X.shape)
"""),
code("""
# one image per activity (dataset 1, fixed subject/rep for a fair comparison)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
rows = rp.build_index([1])
for act in range(1, 7):
    r = [x for x in rows if x["activity"] == act and x["subject"] == 36
         and x["rep"] == 1][0]
    sp, vel, tt = rp.compute_spectrogram(r["path"])
    ax = axes[(act - 1) // 3, (act - 1) % 3]
    ax.imshow(rp.db_norm(rp.to_image(sp, vel)), origin="lower", cmap="jet",
              aspect="auto", extent=[0, tt[-1], -rp.VEL_CROP, rp.VEL_CROP])
    ax.set(title=f"{act} {rp.ACTIVITY_LONG[act]}", xlabel="time [s]",
           ylabel="velocity [m/s]")
plt.tight_layout(); plt.show()
"""),
md("""
The six signatures are visually distinct: walking is sustained and periodic; sit,
stand, pick, drink and fall are transients. Falling shows the strongest, broadest
Doppler burst.

## Is the fixed range window valid everywhere?

The window 10..30 is from the lab example. Care-home subjects may stand at a
different range, which would move their return out of the window.
"""),
code("""
def range_energy(path):
    fc, Ts, nts, Bw, d = rp.read_dat(path)
    ncl = len(d) // nts
    t = np.fft.fftshift(np.fft.fft(d.reshape(nts, ncl, order="F"), axis=0), axes=0)
    dr = t[nts // 2:nts, :]
    nsl = rp._odd_minus_one(ncl)
    bb, aa = butter(4, rp.MTI_CUTOFF, "high")
    m = np.empty((dr.shape[0], nsl), dtype=np.complex128)
    for k in range(dr.shape[0]):
        m[k, :] = lfilter(bb, aa, dr[k, :nsl])
    return np.abs(m[1:, :]).sum(axis=1)

rng = np.random.default_rng(0)
fracs = []
for d in range(1, 8):
    rws = rp.build_index([d])
    sel = rng.choice(len(rws), size=min(25, len(rws)), replace=False)
    prof = np.stack([range_energy(rws[i]["path"]) for i in sel])
    prof = (prof / prof.max(axis=1, keepdims=True)).mean(axis=0)
    fracs.append(prof[rp.RANGE_BIN_LO:rp.RANGE_BIN_HI + 1].sum() / prof.sum())

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(1, 8), fracs, color=["steelblue"]*5 + ["indianred"]*2)
ax.axhline(0.5, ls="--", c="gray")
ax.set(xlabel="dataset", ylabel="energy fraction in window",
       title="Return captured by the fixed range window 10..30")
plt.tight_layout(); plt.show()
print({d: round(f, 2) for d, f in zip(range(1, 8), fracs)})
"""),
md("""
The window captures 0.68-0.75 of the return for the lab datasets (1-5) but only
0.39 (NG Homes) and 0.54 (West Cumbria): the care-home subjects stand closer to
the radar, so part of their return falls below the window. This geometry shift is
one measured cause of the generalization gap in notebook 05.
"""),
])

# --------------------------------------------------------------------------- #
# 03 - classical baseline
# --------------------------------------------------------------------------- #
build("03_baseline_classical.ipynb", [
md("""
# 03 - Classical baseline: physical features + SVM

A classifier built from interpretable micro-Doppler descriptors. Each
spectrogram column is treated as an energy distribution over velocity; we
summarize the Doppler centroid and bandwidth trajectories, the torso/limb and
positive/negative Doppler energy split, and the temporal envelope
(`radar_pipeline.extract_features`, 26 features). Evaluation is
subject-independent (GroupKFold on the (dataset, subject) key).
"""),
code("""
import numpy as np
import matplotlib.pyplot as plt
import joblib
import radar_pipeline as rp
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, accuracy_score

X, meta = rp.load_cache([1])           # dataset 1: balanced, single site
y = meta["activity"] - 1
groups = meta["group"]
F = rp.feature_matrix(X)
print(f"images {X.shape}, features {F.shape}, subjects {len(set(groups))}")
print("feature names:", rp.FEATURE_NAMES)
"""),
md("## Subject-independent cross-validation"),
code("""
models = {
    "SVM (RBF)": make_pipeline(StandardScaler(),
                               SVC(C=10, gamma="scale", random_state=0)),
    "RandomForest": RandomForestClassifier(n_estimators=400, random_state=0,
                                           n_jobs=-1),
}
gkf = GroupKFold(n_splits=5)
results = {}
for name, mk in models.items():
    accs, yt, yp = [], [], []
    for tr, te in gkf.split(F, y, groups):
        m = clone(mk).fit(F[tr], y[tr])
        pred = m.predict(F[te])
        accs.append(accuracy_score(y[te], pred)); yt.append(y[te]); yp.append(pred)
    results[name] = (np.array(accs), np.concatenate(yt), np.concatenate(yp))
    print(f"{name:14s} acc = {np.mean(accs):.3f} +- {np.std(accs):.3f}")
"""),
code("""
yt, yp = results["SVM (RBF)"][1], results["SVM (RBF)"][2]
cm = confusion_matrix(yt, yp)
names = [rp.ACTIVITY_NAMES[i + 1] for i in range(6)]
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(6), names, rotation=45); ax.set_yticks(range(6), names)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set(xlabel="predicted", ylabel="true", title="SVM confusion (subject-independent)")
plt.tight_layout(); plt.show()
"""),
md("""
## What the features see

RandomForest importances show which descriptors carry the signal, which makes the
model defensible rather than a black box.
"""),
code("""
rf = clone(models["RandomForest"]).fit(F, y)
imp = rf.feature_importances_
order = np.argsort(imp)[::-1][:12]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh([rp.FEATURE_NAMES[i] for i in order][::-1], imp[order][::-1],
        color="seagreen")
ax.set(title="Top feature importances (RandomForest)", xlabel="importance")
plt.tight_layout(); plt.show()
"""),
md("""
## Reading the result

- SVM reaches ~0.96 subject-independent on this clean, balanced single-site data.
- Walking is essentially perfect (sustained periodic signature). The dominant
  error is **pick vs drink**: both are low-velocity arm/torso motions with similar
  Doppler extent. The sit/stand pair, near mirror images in Doppler, is largely
  resolved because the positive/negative-Doppler features capture motion
  direction.

The final model is refit on all of dataset 1 and saved.
"""),
code("""
final_svm = clone(models["SVM (RBF)"]).fit(F, y)
joblib.dump(final_svm, "../models/svm_ds1.joblib")
print("saved ../models/svm_ds1.joblib")
"""),
])


# --------------------------------------------------------------------------- #
# 04 - CNN comparison
# --------------------------------------------------------------------------- #
build("04_cnn_comparison.ipynb", [
md("""
# 04 - Does a CNN beat the handcrafted features?

A transfer-learned ResNet18 reads the dB spectrogram directly (first conv reduced
to one channel, pretrained weights averaged). It is fine-tuned on MPS. The
question is whether learned features beat the interpretable ones on the same
subject-independent splits. Epoch selection uses an inner subject-held-out
validation split, so the test fold is never used for model selection.
"""),
code("""
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import radar_pipeline as rp
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score

X, meta = rp.load_cache([1])
y = (meta["activity"] - 1).astype(np.int64)
groups = meta["group"]
Xdb = rp.db_stack(X)
print("device:", rp.get_device(), "| images:", Xdb.shape)
"""),
code("""
gkf = GroupKFold(n_splits=5)
accs, yt, yp = [], [], []
t0 = time.time()
for k, (tr, te) in enumerate(gkf.split(Xdb, y, groups)):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    itr, iva = next(gss.split(Xdb[tr], y[tr], groups[tr]))
    net = rp.make_cnn(6, pretrained=True)
    net, best = rp.train_cnn(net, Xdb[tr][itr], y[tr][itr],
                             Xdb[tr][iva], y[tr][iva],
                             epochs=20, lr=5e-4, batch=32, seed=0)
    pred = rp.cnn_predict(net, Xdb[te])
    accs.append(accuracy_score(y[te], pred)); yt.append(y[te]); yp.append(pred)
    print(f"fold {k+1}: test acc {accs[-1]:.3f}")
accs = np.array(accs)
print(f"\\nCNN acc = {accs.mean():.3f} +- {accs.std():.3f}  "
      f"({time.time()-t0:.0f}s for 5 folds)")
"""),
code("""
cm = confusion_matrix(np.concatenate(yt), np.concatenate(yp))
names = [rp.ACTIVITY_NAMES[i + 1] for i in range(6)]
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(6), names, rotation=45); ax.set_yticks(range(6), names)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set(xlabel="predicted", ylabel="true", title="CNN confusion (subject-independent)")
plt.tight_layout(); plt.show()
"""),
md("""
## Comparison

| model | acc (ds1, subj-indep) | train cost | interpretable |
|-------|----------------------|-----------|---------------|
| features + SVM | 0.96 +- 0.01 | ~1 s | yes (named descriptors) |
| ResNet18 (fine-tuned) | ~0.96 +- 0.03 | ~45 s on MPS | no |

On clean single-site data the CNN does **not** beat the handcrafted features; it
matches them within run-to-run noise, at higher variance and far higher cost,
with the same pick/drink confusion. (CNN figures move a couple of points between
runs from GPU nondeterminism; the conclusion does not.) The interesting question
is whether the extra capacity helps when the test population differs from
training (notebook 05). The final CNN is trained on all of dataset 1 and saved.
"""),
code("""
net = rp.make_cnn(6, pretrained=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
itr, iva = next(gss.split(Xdb, y, groups))
net, best = rp.train_cnn(net, Xdb[itr], y[itr], Xdb[iva], y[iva],
                         epochs=20, lr=5e-4, batch=32, seed=0)
torch.save(net.state_dict(), "../models/cnn_ds1.pt")
print(f"saved ../models/cnn_ds1.pt (inner-val acc {best:.3f})")
"""),
])


# --------------------------------------------------------------------------- #
# 05 - generalization
# --------------------------------------------------------------------------- #
build("05_generalization.ipynb", [
md("""
# 05 - Generalization: lab to elderly care

The headline test. Train on the lab/university datasets (1-5, mostly young
adults), test on the elderly care datasets (6-7, ages up to 98). Fall is dropped
because the target sites barely recorded it, leaving a 5-class problem
(walk, sit, stand, pick, drink). We compare the within-source ceiling to the
cross-domain accuracy for both models.
"""),
code("""
import numpy as np
import matplotlib.pyplot as plt
import joblib
import radar_pipeline as rp
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, accuracy_score

SOURCE, TARGET, CLASSES = [1, 2, 3, 4, 5], [6, 7], [1, 2, 3, 4, 5]
NAMES = [rp.ACTIVITY_NAMES[c] for c in CLASSES]

def load(dsets):
    X, meta = rp.load_cache(dsets)
    keep = np.isin(meta["activity"], CLASSES)
    return (X[keep], (meta["activity"][keep] - 1).astype(np.int64),
            meta["group"][keep])

Xs, ys, gs = load(SOURCE)
Xt, yt, gt = load(TARGET)
Fs, Ft = rp.feature_matrix(Xs), rp.feature_matrix(Xt)
Xs_db, Xt_db = rp.db_stack(Xs), rp.db_stack(Xt)
print(f"source {Xs.shape[0]} files / {len(set(gs))} subjects | "
      f"target {Xt.shape[0]} files / {len(set(gt))} subjects")
"""),
md("## Within-source ceiling (subject-independent CV on the lab pool)"),
code("""
svm = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=0))
gkf = GroupKFold(n_splits=5)
cv = [accuracy_score(ys[te], clone(svm).fit(Fs[tr], ys[tr]).predict(Fs[te]))
      for tr, te in gkf.split(Fs, ys, gs)]
print(f"SVM within-source CV: {np.mean(cv):.3f} +- {np.std(cv):.3f}")
"""),
md("## Cross-domain: train on lab, test on care"),
code("""
svm_full = clone(svm).fit(Fs, ys)
svm_pred = svm_full.predict(Ft)
svm_acc = accuracy_score(yt, svm_pred)

# The CNN result is averaged over seeds: a single fine-tune is noisy on MPS and
# the cross-domain number swings several points run to run, so one seed is not a
# fair comparison.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
itr, iva = next(gss.split(Xs_db, ys, gs))
cnn_accs, cnn_pred = [], None
for seed in range(3):
    net = rp.make_cnn(len(CLASSES), pretrained=True)
    net, _ = rp.train_cnn(net, Xs_db[itr], ys[itr], Xs_db[iva], ys[iva],
                          epochs=20, lr=5e-4, batch=32, seed=seed)
    p = rp.cnn_predict(net, Xt_db)
    cnn_accs.append(accuracy_score(yt, p))
    if cnn_pred is None:
        cnn_pred = p           # seed 0, used for the confusion matrix
cnn_acc = float(np.mean(cnn_accs))
print(f"SVM source->target: {svm_acc:.3f}")
print(f"CNN source->target: {cnn_acc:.3f} +- {np.std(cnn_accs):.3f} "
      f"(seeds: {[round(a, 3) for a in cnn_accs]})")
"""),
code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a_, pred, ttl in [(ax[0], svm_pred, f"SVM target ({svm_acc:.2f})"),
                      (ax[1], cnn_pred, f"CNN target (seed 0)")]:
    cm = confusion_matrix(yt, pred)
    a_.imshow(cm, cmap="Blues")
    a_.set_xticks(range(5), NAMES, rotation=45); a_.set_yticks(range(5), NAMES)
    for i in range(5):
        for j in range(5):
            a_.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    a_.set(xlabel="predicted", ylabel="true", title=ttl)
plt.tight_layout(); plt.show()
"""),
code("""
fig, ax = plt.subplots(figsize=(5, 3))
accs = []
for d in TARGET:
    Xd, yd, _ = load([d])
    accs.append(accuracy_score(yd, svm_full.predict(rp.feature_matrix(Xd))))
ax.bar([rp.DATASET_NAMES[d].split(" Dataset")[0] for d in TARGET], accs,
       color="indianred")
ax.axhline(np.mean(cv), ls="--", c="gray", label="within-source CV")
ax.set(ylabel="SVM accuracy", title="Per target dataset", ylim=(0, 1)); ax.legend()
plt.tight_layout(); plt.show()
print({d: round(a, 3) for d, a in zip(TARGET, accs)})
"""),
md("""
## Conclusions

- SVM accuracy falls from ~0.89 within the lab pool to ~0.77 on the care sites:
  an honest ~12 point generalization gap, larger (~0.74) for West Cumbria.
- The CNN does **not** reliably generalize better than the SVM. Averaged over
  seeds it lands near the SVM (~0.77) with several points of run-to-run swing,
  while the SVM is deterministic. Extra model capacity buys no cross-domain
  robustness here, and its instability is itself a reason to prefer the simpler
  model; both are limited by the domain shift, not by model expressiveness.
- The errors are physical and concentrated in **pick vs drink**: in the elderly
  population these arm motions are slower and smaller, so their already-similar
  low-velocity Doppler signatures overlap further.
- Part of the gap is a **geometry shift** (notebook 02): the fixed range window,
  set on the lab data, captures only ~0.4-0.5 of the care-home return because
  those subjects stand closer to the radar.

Limitations and next steps: re-centre the range window per site, add a few care
recordings to training (domain adaptation), and report fall separately since the
care sites barely recorded it. The lab-only model is saved for reuse.
"""),
code("""
joblib.dump(svm_full, "../models/svm_lab.joblib")
print("saved ../models/svm_lab.joblib")
"""),
])

print("done")
