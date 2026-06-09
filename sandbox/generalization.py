"""
Phase 5 (headline): train on the lab/university datasets, test on the elderly
care datasets. 5-class task (fall dropped: absent from the target). Reports the
within-source subject-independent CV accuracy (the ceiling) and the cross-domain
accuracy for both the classical SVM and the CNN, plus target confusion matrices.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import radar_pipeline as rp

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import clone

SOURCE = [1, 2, 3, 4, 5]      # lab / university
TARGET = [6, 7]               # elderly care
CLASSES = [1, 2, 3, 4, 5]     # drop fall (absent in target)
NAMES = [rp.ACTIVITY_NAMES[c] for c in CLASSES]


def load(dsets):
    X, meta = rp.load_cache(dsets)
    keep = np.isin(meta["activity"], CLASSES)
    X = X[keep]
    y = (meta["activity"][keep] - 1).astype(np.int64)   # 0..4 (classes 1..5)
    g = meta["group"][keep]
    return X, y, g

Xs, ys, gs = load(SOURCE)
Xt, yt, gt = load(TARGET)
print(f"source: {Xs.shape[0]} files, {len(set(gs))} subjects")
print(f"target: {Xt.shape[0]} files, {len(set(gt))} subjects")

Fs, Ft = rp.feature_matrix(Xs), rp.feature_matrix(Xt)
Xs_db, Xt_db = rp.db_stack(Xs), rp.db_stack(Xt)

svm = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=0))

# ---- within-source CV ceiling (subject-independent) ----
gkf = GroupKFold(n_splits=5)
svm_cv = []
for tr, te in gkf.split(Fs, ys, gs):
    m = clone(svm).fit(Fs[tr], ys[tr])
    svm_cv.append(accuracy_score(ys[te], m.predict(Fs[te])))
print(f"\nSVM within-source CV: {np.mean(svm_cv):.3f} +- {np.std(svm_cv):.3f}")

# ---- SVM cross-domain ----
svm_full = clone(svm).fit(Fs, ys)
svm_pred = svm_full.predict(Ft)
print(f"SVM source->target: {accuracy_score(yt, svm_pred):.3f}")
print("target confusion (SVM), rows=true cols=pred", NAMES)
print(confusion_matrix(yt, svm_pred))

# ---- CNN cross-domain (inner-val epoch selection on source subjects) ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
itr, iva = next(gss.split(Xs_db, ys, gs))
net = rp.make_cnn(len(CLASSES), pretrained=True)
net, best = rp.train_cnn(net, Xs_db[itr], ys[itr], Xs_db[iva], ys[iva],
                         epochs=20, lr=5e-4, batch=32, seed=0)
cnn_pred = rp.cnn_predict(net, Xt_db)
print(f"\nCNN inner-val best: {best:.3f}")
print(f"CNN source->target: {accuracy_score(yt, cnn_pred):.3f}")
print("target confusion (CNN), rows=true cols=pred", NAMES)
print(confusion_matrix(yt, cnn_pred))

# per-target-dataset breakdown for the SVM
print("\nper-target-dataset accuracy:")
for d in TARGET:
    Xd, yd, _ = load([d])
    Fd = rp.feature_matrix(Xd)
    print(f"  dataset {d}: SVM {accuracy_score(yd, svm_full.predict(Fd)):.3f}")
