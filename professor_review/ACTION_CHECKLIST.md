# Action checklist before submission

Ordered by leverage. P0 = do before you submit anything. P1 = high value if you have a
few hours. P2 = nice to have.

---

## P0 — Submission hygiene (the scaffolding is currently tracked and WILL ship)

Right now `git ls-files` includes the AI-agent scaffolding. If you submit by pushing
or zipping the git repo, the professor receives all of it. Verify and fix:

```bash
git ls-files | grep -E '^(context/|sandbox/|PROJECT_BRIEF)'
```

If that prints anything, those files are in the submission. Decide your submission
method:

- **Submitting a clean export (recommended):** build the hand-off from the intended
  set only — `README.md ONE_PAGER.md PRESENTATION.md notebooks/ models/` (and
  `data/` only if the course wants the dataset back — usually it does **not**; they
  gave it to you). Do not include `context/`, `sandbox/`, `slides/`, `PROJECT_BRIEF.md`,
  the `.pptx`, `cache/`, `.venv/`, or `professor_review/`.

- **Submitting the git repo:** untrack the scaffolding first and add it to
  `.gitignore`:
  ```bash
  git rm -r --cached context sandbox PROJECT_BRIEF.md
  printf '\ncontext/\nsandbox/\nPROJECT_BRIEF.md\nprofessor_review/\n' >> .gitignore
  git commit -m "Remove internal scaffolding from submission"
  ```

**Specifically make sure these never reach the examiner:**
- `sandbox/build_notebooks.py` — proves the notebooks were generated, not written.
- `PROJECT_BRIEF.md` — opens "A fresh session should read this file…", an agent prompt.
- `context/decisions_log.md` — also holds slightly older CNN numbers (0.961/0.777) that
  differ from the notebooks (0.964/0.761); don't give anyone two versions to diff.
- `professor_review/` — this folder. Do not ship my notes.

## P0 — Confirm the deliverable stands alone without the cache/data

The notebooks import `radar_pipeline` and read `cache/` + `data/` (both gitignored).
Make sure whoever runs them can rebuild: `notebooks/README.md` documents the setup and
nb02 rebuilds the cache. Sanity-check on a clean checkout that `Restart & Run All`
works end-to-end with only the submitted files + the provided dataset. If the grader
won't run code, confirm every notebook's outputs are saved (they are) so the figures
render without execution.

---

## P1 — Strengthen against the obvious defense questions (see REVIEW.md)

1. **Run the range-window fix you proposed (REVIEW item B).** Highest-value addition.
   Add a cell to nb05 (or nb02) that re-centres the range window for ds6/ds7 (per-site
   or energy-peak-centred) and reports whether cross-domain accuracy moves from ~0.77.
   Turns your best "future work" line into a result and pre-empts the #1 question.

2. **Defuse the `C=10` question (REVIEW item A).** Add one small cell showing SVM
   accuracy is flat across `C ∈ {1, 10, 100}` on the subject-independent CV, or a
   sentence stating it was fixed in exploration and not re-tuned per fold.

3. **Add a `classification_report` (REVIEW item F).** 3 lines in nb03 and nb05 so the
   per-class precision/recall behind "pick recall ~0.88" is visible in the deliverable
   itself, not only in the unsubmitted log.

4. **One numeric port-validation in nb02 (REVIEW item E).** A single assertion or
   printed correlation that the Python spectrogram matches the MATLAB reference, so the
   port rests on a number and not only on a visual.

## P1 — Make the notebooks read as human-authored (see AUTHENTICITY.md)

- Edit each notebook by hand after generation; the P1 additions above double as this.
- Re-run interactively so execution counts reflect real use, not one clean pass.
- Add a couple of genuine first-person notes (what you tried and dropped, and why).

---

## P2 — Polish

- Decide whether `data/` (1754 `.dat` + the consent/info PDFs) should be in the
  submission at all. The dataset was provided to you; re-submitting ~hundreds of MB is
  usually unwanted. Confirm against the assignment instructions.
- `models/cnn_ds1.pt` is 44 MB; fine to include, but check any submission size cap.
- Confirm the `.pptx` vs `PRESENTATION.md`: the README points to the markdown outline;
  make sure the actual slide deck you present from matches it and that the figures it
  references (nb01 count table, nb02 DSP/range figures, nb03 confusion, nb04 table,
  nb05 bar + confusion) are exported and embedded.

---

## Quick reference: what should be in the submission

```
README.md
ONE_PAGER.md
PRESENTATION.md            (+ the actual slide deck you present)
notebooks/  01..05 .ipynb, README.md, radar_pipeline.py
models/     svm_ds1.joblib, cnn_ds1.pt, svm_lab.joblib
data/       ONLY if the course asks for the dataset back (likely not)
```

What must **not** be in it: `context/`, `sandbox/`, `PROJECT_BRIEF.md`, `slides/`,
`*.pptx` source if separate, `cache/`, `.venv/`, `professor_review/`.
