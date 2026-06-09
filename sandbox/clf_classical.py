"""
Phase 3/4 (classical): subject-independent CV on dataset 1 using physical
micro-Doppler features + a classical model. Compares RandomForest and an RBF SVM.
Reports per-fold and pooled accuracy and a pooled confusion matrix.
"""
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import radar_pipeline as rp

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score

X, meta = rp.load_cache([1])
y = meta["activity"] - 1                 # 0..5
groups = meta["group"]
F = rp.feature_matrix(X)
print(f"X={X.shape}  features={F.shape}  subjects={len(set(groups))}")

models = {
    "rf": RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
    "svm": make_pipeline(StandardScaler(),
                         SVC(C=10, gamma="scale", random_state=0)),
}

gkf = GroupKFold(n_splits=5)
for name, mk in models.items():
    accs, y_true_all, y_pred_all = [], [], []
    for tr, te in gkf.split(F, y, groups):
        from sklearn.base import clone
        m = clone(mk)
        m.fit(F[tr], y[tr])
        pred = m.predict(F[te])
        accs.append(accuracy_score(y[te], pred))
        y_true_all.append(y[te]); y_pred_all.append(pred)
    yt = np.concatenate(y_true_all); yp = np.concatenate(y_pred_all)
    cm = confusion_matrix(yt, yp)
    print(f"\n=== {name} === per-fold acc: {np.round(accs,3)}  "
          f"mean={np.mean(accs):.3f}+-{np.std(accs):.3f}")
    print("pooled confusion matrix (rows=true, cols=pred), classes "
          f"{[rp.ACTIVITY_NAMES[i+1] for i in range(6)]}:")
    print(cm)
    per_class = cm.diagonal() / cm.sum(axis=1)
    for i in range(6):
        print(f"  {rp.ACTIVITY_NAMES[i+1]:6s} recall={per_class[i]:.2f}")
