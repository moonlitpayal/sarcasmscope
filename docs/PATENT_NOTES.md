# Patent notes (background only; not legal advice)

Talk to your institute's IPR/innovation cell and a registered patent agent before filing or publishing anything.

## Honest starting point
* A detection algorithm, a model architecture or a benchmark is usually NOT patentable "per se" under Indian law
  (Section 3(k) excludes computer programmes, mathematical methods and algorithms as such). Software-related
  inventions can be considered when they produce a concrete technical effect or solve a technical problem in a
  specific way. Whether anything here qualifies is a question for a patent agent after a prior-art search.
* Recent papers already cover multilingual Indic sarcasm detection with incongruity and emoji signals (see the
  literature workbook), so a broad claim on "multilingual code-mixed sarcasm detection" is unlikely to be new.
* Open-sourcing the code or publishing a paper BEFORE filing can destroy novelty. Keep the repository private for now.

## Candidate subject matter to discuss with the agent (none is promised to be patentable)
1. A reliability pipeline that combines cluster-based leakage-safe splitting, calibration and a validated abstention
   threshold so that a deployed detector refuses to answer when its confidence is below a threshold chosen on held-out data.
2. A script-aware robustness/perturbation check run at inference time (the `/robustness` endpoint) that flags unstable predictions.
3. Any new technical component you build and can show, with measurements, to improve a technical metric
   (accuracy, latency, memory, robustness) over the best prior method.

## What you need before a useful agent meeting
* A dated, signed invention disclosure (problem, solution, drawings, an example, measured technical effect).
* A prior-art search log (Google Patents, Espacenet, Indian Patent Office search) with the closest documents.
* A list of every contributor and what each one invented; your institute's IP policy.

## Order of operations
Prior-art search -> invention disclosure -> IPR cell meeting -> (provisional filing) -> then paper submission/code release.
