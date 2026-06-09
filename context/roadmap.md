# Roadmap

Provisional. Each step ends in a finding that justifies the next. Update as we
go. "Done" means observed and logged in `decisions_log.md`.

Status: Phases 0-6 complete. Sandbox scripts settled every choice (D1-D14); the
five notebooks, the saved models, the one-pager and the presentation outline are
built from them. See `decisions_log.md` for the evidence behind each step.

## Phase 0 — Foundations  (done)
- [x] Read README, MATLAB example, label extractor, course slide deck.
- [x] Confirm `.dat` format and the physical meaning of the samples.
- [x] Verify the Python DSP port against MATLAB on one file.
- [x] Catch the data gotchas (filename padding, subject-ID collision, label
      naming contradiction).

## Phase 1 — Build the dataset index and labels
- Parse every filename → (dataset, subject, activity, repetition, path).
- Report per-class and per-dataset counts; check imbalance.
- Implement (dataset, subject)-keyed grouping for splitting.

## Phase 2 — Batch preprocessing + caching
- Run the verified DSP over a chosen dataset list; cache spectrograms (`.npy`)
  + a metadata table. One config switch controls which datasets are included.
- Sanity-check: plot one spectrogram per activity; confirm classes look
  visually distinct and the range-bin window is valid.

## Phase 3 — First classifier on December 2017 (baseline)
- Decide input representation (crop/normalization) from the plots.
- Train a first model with subject-independent CV; record accuracy and a
  confusion matrix. Keep it simple; this is the reference point.

## Phase 4 — Improve and compare
- Add a second approach (the find from Phase 3 decides which: classical features
  + classical model, and/or a fine-tuned CNN on MPS). Compare honestly.
- Error analysis from the confusion matrix tied to physics (which activities
  mirror each other in Doppler and why).

## Phase 5 — Generalization
- Scale to more datasets. Test train-on-young/lab → test-on-elderly/care.
- Report what holds and what breaks; this is the headline result for
  interpretation and originality.

## Phase 6 — Deliverables
- Build the clean professor-facing **notebooks** from the settled sandbox code.
- Write the **1-pager** and the **9-minute presentation** outline, every claim
  traceable to `decisions_log.md`. Plain, human prose — no AI tells.
