# Notebooks

Professor-facing build. Run in order; each notebook is self-contained and
documents the choices it makes. Shared, verified code lives in
`radar_pipeline.py` (imported by every notebook) so the DSP parameters, file
parsing, and subject-independent splitting are defined once.

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | `01_dataset_index.ipynb` | Parse filenames, class/site distribution, the label and subject-ID traps, the `(dataset, subject)` split key. |
| 02 | `02_preprocessing.ipynb` | The FMCW DSP chain step by step (range FFT, MTI, slow-time STFT), the model input, caching all files, and the range-window check. |
| 03 | `03_baseline_classical.ipynb` | Physical micro-Doppler features + SVM/RandomForest, subject-independent CV, confusion matrix, feature importances. Saves `../models/svm_ds1.joblib`. |
| 04 | `04_cnn_comparison.ipynb` | Transfer-learned ResNet18 on the dB images, same splits, honest comparison to the classical model. Saves `../models/cnn_ds1.pt`. |
| 05 | `05_generalization.ipynb` | The headline: train on lab/university, test on elderly care. Saves `../models/svm_lab.joblib`. |

## Setup
```
python3.13 -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-learn matplotlib pillow tqdm torch torchvision \
            jupyter ipykernel pandas
python -m ipykernel install --user --name radar --display-name "Python (radar)"
```
Run the notebooks with the `Python (radar)` kernel. Notebook 02 caches all 1754
spectrograms to `../cache/` on first run (a few minutes, parallel across cores)
and is idempotent; the modeling notebooks load the cache in seconds. The CNN
trains on the Apple-silicon GPU (MPS) in under a minute per fit; its accuracy
varies a couple of points between runs, which the text accounts for.

## Reproducibility note
Classical results (SVM, RandomForest) are deterministic. CNN results use fixed
seeds but MPS is not bit-reproducible, so notebook 05 averages the CNN over three
seeds for a fair cross-domain comparison.
