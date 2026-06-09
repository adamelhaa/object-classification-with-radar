# Professor Review — Human Activity Classification with FMCW Radar

*Reviewing only the intended deliverable: `README.md`, `notebooks/`, `models/`, `data/`,
plus the top-level `ONE_PAGER.md` / `PRESENTATION.md` the README points to. Internal
folders (`context/`, `sandbox/`, `slides/`, `PROJECT_BRIEF.md`) are treated as NOT
submitted — see `ACTION_CHECKLIST.md`, this is not currently true in git.*

## Provisional verdict: **Pass, comfortably — likely a strong mark.**

If I received this as the examiner for EE4775, I would pass it without hesitation and
probably grade it well above the median. The methodology is correct where it most
often goes wrong in this course, the narrative is honest, and the headline result is
a genuine, evidence-backed finding rather than a leaderboard number. The risks below
are about *defensibility under questioning* and *authenticity optics*, not about
whether the work is sound.

---

## What earns the grade

**1. The evaluation is subject-independent and keyed correctly.** This is the single
most common failure mode in this assignment and the work gets it exactly right:
splits hold out whole `(dataset, subject)` pairs, and the notebook *demonstrates* why
(34 bare subject IDs collide across campaigns). Most submissions silently leak the
same person into train and test and report an inflated ~0.99. This one does not, and
it makes the point explicitly (nb01, D3/D8). This alone clears the "critical
attitude" bar.

**2. The DSP chain is a faithful, step-by-step port of the provided MATLAB**, walked
through on a real walking file (range FFT → MTI high-pass → slow-time STFT, nb02),
with the velocity conversion `v = f_d·c/(2·fc)` shown. The single shared
implementation in `radar_pipeline.py` (imported by every notebook) is good
engineering — the DSP parameters cannot drift between notebooks.

**3. The data was genuinely interrogated, not assumed.** Three findings prove real
engagement with the bytes:
   - 3 filename typos found by cross-checking the leading digit `K` against the `Ayy`
     field (`2P34A03R1`, `3P03A02R1`, `6P01A05R03`).
   - The README-vs-`Label_extract4.m` disagreement on activities 4/5 is caught and a
     side is chosen with a stated reason.
   - The fall-class coverage is found to be the **reverse of the project brief** (brief
     said ds6 lacks fall / ds7 has it; the data shows ds7 has **zero** falls, ds6 has 3)
     — and the work corrects the brief from the data. Examiners notice this kind of thing.

**4. The model comparison is honest and the right shape.** Interpretable physical
features + RBF SVM (~0.96) vs a fine-tuned 1-channel ResNet18 (~0.96) on identical
subject-independent splits. The conclusion — capacity is *not* the bottleneck on
clean single-site data, so the simple interpretable model wins on cost and
defensibility — is the correct reading and is argued from the confusion structure
(pick↔drink overlap; sit/stand resolved by directional Doppler), tying conclusions to
physics.

**5. The headline is a real result, reported against itself.** Train lab (ds1–5),
test elderly care (ds6–7), 5-class: within-source CV ~0.89 → cross-domain ~0.77
(~0.74 West Cumbria). The CNN's extra capacity buys no cross-domain robustness and is
*less* stable (seed swing 0.71–0.82). Crucially, a **second, measured cause** is
given beyond "domain shift": the fixed range window captures only ~0.39–0.54 of the
care-home return vs 0.68–0.75 in the lab, because elderly subjects stand closer — a
geometry shift quantified in nb02. Diagnosing *why* the model fails, with a number, is
exactly the "interpretation" the rubric rewards.

**6. Reproducibility is handled like an adult.** Classical results deterministic; CNN
non-bit-reproducible on MPS, so the cross-domain number is averaged over 3 seeds and
the variance is reported rather than hidden. Inner subject-held-out validation split
for CNN epoch selection means the test fold is never used for model selection (nb04).

---

## Where a sharp examiner will push (and how exposed you are)

Ordered by how likely it is to come up in a defense. None are fatal; have an answer.

**A. "Where did `C=10` for the SVM come from?"** — *Medium likelihood.* The RBF
`C=10, gamma=scale` appears with no tuning shown in the deliverable. If it was tuned
on the same CV you report, that is mild optimism. Best answer: state it was fixed
from early exploration on a held-out subject and is not re-tuned per fold; or add one
sentence/cell showing the result is flat across `C ∈ {1,10,100}`. Cheap to defuse.

**B. "You diagnosed the range-window geometry shift — did you try the fix?"** —
*High likelihood, highest-value gap.* You identify re-centering the range window per
site as the fix and quantify the problem beautifully, but you never run the
experiment, even once. This is the most natural question in the room and the single
addition that would most strengthen the work: re-run ds6/ds7 with a per-site or
energy-centred window and report whether ~0.77 moves. Even a negative result is a
win. If you don't add it, rehearse: "future work, because it changes the documented
pipeline and we chose to report the provided method faithfully."

**C. The SVM/CNN cross-domain comparison is slightly unequal.** SVM trains on all 970
source files; the CNN trains on 80% (20% held for early-stopping validation). Within
seed noise this doesn't change the conclusion, but if asked, acknowledge it rather
than be caught.

**D. Model comparison rests on dataset 1 only** (360 files, 20 subjects); one CNN fold
scores 1.000 on a small test fold. Fine, but say "single-site, small-sample" out
loud so it doesn't look like over-claiming.

**E. "How do you *know* the Python port matches the MATLAB?"** The deliverable shows a
*visual* match on one walking file ("consistent with", "confirms the port"). For a
radar course a quantitative check (e.g. correlation of the Python vs MATLAB
spectrogram, or of range-profile peaks) would be stronger. You have
`sandbox/validate_pipeline.py` — but it isn't in the submitted set. Consider folding
one numeric assertion into nb02.

**F. Minor reporting gaps.** Confusion matrices are shown but no per-class
precision/recall/F1 table; "pick recall ~0.88" lives only in the (unsubmitted)
decisions log. A 3-line `classification_report` in nb03/nb05 would make the
physical-error argument self-contained.

---

## Internal-consistency check (deliverable only)

Every number a professor can cross-reference inside the submitted set agrees:
counts (312/312/311/311/311/197, fall absent in ds7), 0.961 SVM / 0.964 CNN on ds1,
0.891 within-source, 0.767 SVM / 0.761 CNN cross-domain, 0.789/0.744 per care site,
range fractions 0.39/0.54 vs 0.68–0.75. The narrative docs (README, ONE_PAGER,
PRESENTATION) quote these faithfully. No contradictions a reader could catch.

(For your eyes only: the *unsubmitted* `decisions_log.md` records slightly older CNN
numbers — D11 "0.961", D14 "0.777" vs the notebooks' 0.964 / 0.761 — within seed
noise, but make sure that log never ships, or reconcile it, so the two can't be
diffed against each other. See `ACTION_CHECKLIST.md`.)

## Rubric read (typical axes for this course)

| Axis | Assessment |
|------|-----------|
| Correct DSP / signal understanding | Strong — faithful port, walked step by step |
| Methodology / critical attitude | Strong — subject-independent splitting is the centrepiece |
| Interpretation of results | Strong — errors tied to physics; gap diagnosed with a measured cause |
| Originality | Good — honest CNN-vs-classical and the generalization study are the payload |
| Effort / completeness | High — 1754 files, full pipeline, two models, transfer study |
| Presentation | Strong — but *too* uniform; see `AUTHENTICITY.md` |

Bottom line: the work is a clear pass and a strong one. Spend remaining effort on
(1) submission hygiene so the AI-agent scaffolding never ships, and (2) the
range-window experiment in B — that's the difference between "good" and "they clearly
own this."
