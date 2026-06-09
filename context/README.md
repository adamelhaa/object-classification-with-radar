# Context

This folder is the working specification for the project. It is built up
incrementally from things actually observed in the data and code, not assumed.
The intent: a fresh session can read this folder and execute the full project
with no further questions.

Read in this order:

1. `scope.md` — what the project is, what must be delivered, how it is graded,
   the hard constraints, and how we work.
2. `data.md` — the dataset: file format, what the numbers mean physically,
   per-dataset breakdown, and the gotchas found so far.
3. `dsp_pipeline.md` — the verified signal-processing chain that turns one
   `.dat` file into a micro-Doppler spectrogram, with exact parameters.
4. `roadmap.md` — the current plan. Provisional and updated as findings land.
5. `decisions_log.md` — chronological record of choices and the evidence behind
   each one. Every method in the deliverables must trace back to an entry here.

Two working areas in the repo root:

- `sandbox/` — throwaway `.py` for exploration, plots, and trials. Nothing here
  is a deliverable.
- (later) notebooks — the clean, professor-facing build, produced only once the
  approach is fully settled here.
