# Scope

## Problem
Course EE4775 (Object Classification with Radar, TU Delft, Dr. F. Fioranelli).
Given radar recordings of people performing daily activities, classify each
recording into one of **6 activities**. It is stated as an open problem: build
a classification pipeline that works as well as reasonable and **justify every
choice**. Work is in a group of 3.

Activities (per the README datasheet, which is authoritative):
1 walking, 2 sitting down, 3 standing up, 4 picking up an object,
5 drinking, 6 falling.

## Deliverables
- A **1-page** description of the implementation and ideas.
- **Code** that runs and reproduces results, well documented. For this project
  the code is delivered as **Jupyter notebooks** (user requirement).
- The **trained model(s)**, documented so they can be re-run/checked.
- A **~9-minute presentation** (+ Q&A). Not submitted, but must be ready.

The 1-pager + code + presentation are worth 25% of the course grade.

## Grading (four axes, 25% each)
1. Correct use of radar signal processing & ML methods, with clear why/how.
2. Interpretation of results and critical attitude (conclusions tied to results).
3. Originality of the proposed solution.
4. Presentation: structure, clarity, Q&A.

Target: a comfortable, defensible pass — "good" band — not a state-of-the-art
chase. The professor explicitly says not to aim for perfect accuracy but for
reasonable solutions and meaningful analysis.

## Hard constraints (from the user)
- **Notebooks only** for the professor-facing deliverable. Exploration happens
  in `sandbox/*.py`; the clean notebooks are produced last, once settled.
- **Must not read as AI-generated.** Comments and markdown are terse, plain, and
  human. No filler, no hedging, no decorative bullet lists, no emoji. Explain
  only what isn't obvious from the code.
- **Self-explanatory but not verbose.** A teammate should follow a notebook on
  one read before the presentation and be able to defend every step.
- **Dataset-parameterized.** The pipeline must run on a single dataset first to
  prove it works, then scale to all 7 by changing a config value — no rewrites.
- **Coherent with the course slides** where it is natural (same terminology and
  methods taught in the lectures). Do not force references that don't fit.
- **Everything backed.** No method or number appears in a deliverable unless it
  is justified by something observed and recorded in `decisions_log.md`.

## How we work
Evidence-driven and stepwise. Decide the next step from the results of the last
one; do not pre-commit the whole pipeline. Each decision is logged with the
observation that motivated it.

## Compute
MacBook Pro (Apple silicon). PyTorch MPS backend confirmed working. Budget is
minutes, not hours: preprocessing is cached once; classical models train in
seconds; any CNN fine-tunes on MPS in a few minutes.

## Timing note
The course slides list a submission deadline of "Tuesday 9 June, 23:00" and an
oral presentation on 11 June. Today is 2026-06-08. Confirm whether these dates
apply to the current run; they imply a very short window.
