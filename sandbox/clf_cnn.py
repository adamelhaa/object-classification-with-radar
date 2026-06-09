"""
Phase 3/4 (CNN): subject-independent CV on dataset 1, transfer-learned ResNet18
on dB micro-Doppler images. Epoch selection uses an inner subject-held-out
validation split, so the outer test fold is never used for model selection.
Reports per-fold and pooled accuracy and a pooled confusion matrix.
"""
import sys, time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import radar_pipeline as rp

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score

X, meta = rp.load_cache([1])
y = (meta["activity"] - 1).astype(np.int64)
groups = meta["group"]
Xdb = rp.db_stack(X)
print(f"images={Xdb.shape}  device={rp.get_device()}")

gkf = GroupKFold(n_splits=5)
accs, yt_all, yp_all = [], [], []
t0 = time.time()
for k, (tr, te) in enumerate(gkf.split(Xdb, y, groups)):
    # inner validation split (subject-disjoint) for epoch selection
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    itr, iva = next(gss.split(Xdb[tr], y[tr], groups[tr]))
    net = rp.make_cnn(6, pretrained=True)
    net, best = rp.train_cnn(net, Xdb[tr][itr], y[tr][itr],
                             Xdb[tr][iva], y[tr][iva],
                             epochs=20, lr=5e-4, batch=32, seed=0)
    pred = rp.cnn_predict(net, Xdb[te])
    a = accuracy_score(y[te], pred)
    accs.append(a); yt_all.append(y[te]); yp_all.append(pred)
    print(f"fold {k+1}: inner_best={best:.3f}  test_acc={a:.3f}  "
          f"({time.time()-t0:.0f}s elapsed)")

yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
cm = confusion_matrix(yt, yp)
print(f"\n=== CNN === per-fold acc: {np.round(accs,3)}  "
      f"mean={np.mean(accs):.3f}+-{np.std(accs):.3f}")
print("pooled confusion matrix (rows=true, cols=pred):")
print(cm)
per_class = cm.diagonal() / cm.sum(axis=1)
for i in range(6):
    print(f"  {rp.ACTIVITY_NAMES[i+1]:6s} recall={per_class[i]:.2f}")
