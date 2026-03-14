# Codex Plan

## Goal

Test whether the paper's core conclusions survive two implementation concerns:

1. the perturbation RNG bug
2. the choice of perturbation scale `sigma`

Do this in a way that maps cleanly onto the paper's claims and produces easy-to-read figures.

## Current State

Completed:

1. Repo explored and paper reviewed.
2. `uv` virtual environment created in `.venv`.
3. `countdown` dataset downloaded.
4. GPU launch issue diagnosed:
   - `randopt.py` overrides `CUDA_VISIBLE_DEVICES` unless `--cuda_devices` is passed explicitly.
5. 1-GPU as-is smoke test completed successfully on:
   - model: `Qwen/Qwen2.5-3B-Instruct`
   - task: `countdown`
6. A first A/B was completed for:
   - released sampler (`per_parameter`)
   - fixed RNG sampler (`global_stream`)
7. A first fixed-RNG sigma sweep was completed on the same `3B` setup.

Already learned:

- The RNG bug changes the perturbation reward distribution materially.
- The bug does not obviously destroy `RandOpt`; on the first slice, top-1 got worse but top-4 ensemble improved.
- Sigma matters under the fixed RNG sampler, and very large sigma is destructive.

## What We Are Testing

There are two related but distinct scientific questions.

### Question A: RandOpt robustness

Does `RandOpt` still improve over the base model when the perturbation sampler is corrected?

This maps most closely to:

- the teaser panel on `Countdown`
- the `RandOpt` practical algorithm sections
- the LLM accuracy comparison figures

This is not yet a clean reproduction of the paper's density-scaling figure.

### Question B: Density / thicket structure

Does the local neighborhood still look "dense" with task-improving perturbations under the corrected sampler, and how does that change with model scale?

This maps more closely to:

- the density claims in Section 2
- the scaling-law figure

This is the more direct scientific test of the critique.

### Question C: Perturbation parameterization

Is constant absolute additive noise in raw weight coordinates a sensible neighborhood definition, or are the results sensitive to parameter scale across tensors?

This is a weaker critique than the seed bug, but still scientifically meaningful.

Possible alternative:

- per-tensor relative-norm perturbations, e.g. scale Gaussian noise by each tensor's RMS or Frobenius norm per element

## Experiment Roadmap

### Phase 1: Unified cross-scale absolute-noise sweep

Purpose:

- determine whether the seed bug materially changes `RandOpt` outcomes
- determine whether the local density / hit-rate picture changes under the corrected sampler
- avoid separate duplicate sweeps for `RandOpt`-style and density-style questions

Configuration:

- task: `countdown`
- models:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
- sampler modes:
  - `per_parameter` (released behavior)
  - `global_stream` (fixed RNG)
- use the same:
  - dataset slice
  - global seed
  - population size
  - top-`K`
- run one broader sigma grid that supports both interpretations

Sigma choice for this phase:

- include paper-style `RandOpt` sigmas: `0.001,0.002,0.003`
- include the paper's density-analysis sigma: `0.005`
- optionally include one smaller sigma if needed for stability on smaller models

Sigma protocol for this phase:

- primary runs should use a mixed-sigma population, matching the repo / paper style more closely
- each sampled perturbation should still retain its individual sigma in the saved raw results
- this lets one run support both:
  - paper-faithful mixed-population `RandOpt` evaluation
  - post hoc per-sigma analysis from the same raw samples
- only add separate per-sigma runs if the mixed-population results leave an ambiguity that cannot be resolved from the saved per-sample records

Primary measurements:

1. Base test accuracy
2. `RandOpt` ensemble accuracy at `K=1`, `K=2`, and one larger `K`
3. Mean sampled train reward
4. Max sampled train reward
5. Hit-rate above base-train reward
6. Hit-rate above `base + margin`
7. Full sampled reward distribution summary

Outputs:

- `RandOpt` behavior by model size
- density / hit-rate by model size
- first-pass sigma trends from the same run artifacts

Success criterion:

- enough evidence to say whether the bug materially changes:
  - `RandOpt` behavior
  - density / hit-rate trends across model sizes

### Phase 2: Sigma sensitivity under fixed RNG

Purpose:

- understand whether the corrected sampler prefers a different local noise scale

Configuration:

- fixed RNG only: `global_stream`
- same model family
- sigma sweep centered around:
  - paper-style `RandOpt` values
  - a smaller range if needed

Primary measurements:

1. Mean sampled train reward vs sigma
2. Hit-rate vs sigma
3. Best ensemble accuracy vs sigma

Important note:

- this does not fully resolve the critique's cross-scale calibration issue
- it is still a useful practical check for whether the released sigma choices are unstable under the corrected sampler

### Phase 3: Relative-norm perturbation ablation

Purpose:

- test whether the paper's conclusions depend strongly on using constant absolute additive noise
- check whether a scale-aware perturbation changes the observed density or `RandOpt` behavior

Configuration:

- use the corrected sampler (`global_stream`) as the baseline for comparison
- compare two perturbation families:
  - absolute additive noise
  - per-tensor scale-aware additive noise
- run on the same model family and task setup used in earlier phases

Relative-norm definition for this phase:

- scale each tensor's perturbation by the empirical standard deviation of that tensor's base weights
- intended update form:
  - `std = p_base.std()`
  - `p = p + sign * sigma * std * noise`
- interpret `sigma` as a dimensionless fraction of the tensor's own parameter spread
  - example: `sigma=0.1` means perturb by roughly `10%` of that tensor's standard deviation

Implementation requirements:

- compute the scale from the unperturbed base weights, not from already-perturbed weights
- apply the same scale definition in both:
  - perturb
  - restore
- avoid recomputing from mutated tensors mid-run

Candidate sigma sweep for this phase:

- `0.05, 0.1, 0.2, 0.5`
- goal: bracket the range where perturbations are neither negligible nor clearly destructive

Primary measurements:

1. Mean sampled train reward
2. Hit-rate vs base
3. Best ensemble accuracy
4. Sensitivity to perturbation scale

Interpretation goal:

- determine whether the apparent thicket structure is robust to a more scale-aware perturbation model

## Planned Figures

The final output should be easy to read and should answer one question per figure.

### Figure 1: RandOpt A/B by model size

Question:

- does fixing the RNG bug change `RandOpt` performance?

Plot:

- x-axis: model size
- y-axis: test accuracy
- separate series or grouped bars for:
  - base
  - released sampler
  - fixed RNG sampler
- separate panels or markers for `K=1`, `K=2`, and larger `K`

### Figure 2: Hit-rate / density by model size

Question:

- does fixing the RNG bug change the apparent local density of improving perturbations?

Plot:

- x-axis: model size
- y-axis: hit-rate
- separate lines for:
  - released sampler
  - fixed RNG sampler

Possible variants:

- hit-rate above base
- hit-rate above `base + 0.05`

### Figure 3: Sigma sensitivity under fixed RNG

Question:

- what sigma values work best once the sampler is corrected?

Plot:

- x-axis: sigma
- y-axis: one of:
  - mean sampled train reward
  - hit-rate
  - best ensemble accuracy
- separate series for each model size

### Figure 4: Reward distribution comparison

Question:

- how does the shape of the perturbation distribution change?

Plot:

- violin, ridge, or histogram plots
- compare released vs fixed RNG
- one row per model size

### Figure 5: Absolute vs Relative-Norm Perturbations

Question:

- do the main effects survive when perturbations are scaled relative to tensor magnitude?

Plot:

- x-axis: model size
- y-axis: hit-rate or ensemble accuracy
- lines for:
  - released absolute-noise sampler
  - fixed absolute-noise sampler
  - fixed relative-norm sampler

## Execution Order

1. Lock down the shared protocol on `3B` with the unified mixed-sigma grid.
2. Expand that same cross-scale sweep to `0.5B`, `1.5B`, and `7B`.
3. Build Figure 1 and Figure 2 from those results.
4. Use the same run artifacts to draft sigma trends, then run extra fixed-RNG sigma sweeps only where resolution is missing.
5. Build Figure 3.
6. If useful, add distribution plots as Figure 4.
7. After the seed-bug and sigma questions are settled, run the relative-norm perturbation ablation and build Figure 5.

Parallelization note:

- keep protocol-locking runs simple, usually on 1 GPU
- once the protocol is fixed, use the 3 available GPUs to parallelize independent runs across:
  - model sizes
  - sampler modes
- the goal is to reduce wall-clock time without changing the per-run protocol

## Constraints

- Keep all notes and outputs under `codex/`.
- Favor 1-GPU runs unless parallelism is clearly needed.
- Do not bring in PPO/GRPO/ES yet.
- Keep the figures interpretable and directly tied to one question each.
- Treat relative-norm perturbations as a follow-up robustness test, not a replacement for the paper-faithful absolute-noise experiments.
- Diversity / Spectral Discordance is out of scope for the narrow initial subset.
  - If the density and `RandOpt` results remain interesting after Phases 1-2, add a later multi-task diversity phase.

## Operating Notes

- Andy may be asleep while this work is running.
- Work autonomously and keep moving until the current experiment plan is completed or a real blocker is reached.
- Babysit long runs rather than fire-and-forget them:
  - monitor progress
  - use timed check-ins / sleep intervals
  - verify run completion
  - react to failures quickly
- Stick to the plan unless new evidence clearly justifies a change.

GPU usage policy:

- make active use of free GPUs when this speeds up independent runs
- be extremely careful not to use GPUs that are currently occupied by other users
- treat GPU availability as dynamic and re-check before launching new runs
- current known free set at last inspection:
  - `4`
  - `6`
  - `7`
- do not assume that this set remains valid later; confirm before each batch

## Raw Data Retention

Do not treat plots as the primary artifact.

For each run, keep:

1. full `results.json`
2. full sampled perturbation records
3. run arguments / configuration
4. enough metadata to identify:
   - model
   - task
   - sampler mode
   - sigma grid
   - population size
   - top-`K`
   - seed

Plotting policy:

- plots should be generated from saved raw results, not reconstructed from terminal logs
- avoid deleting raw experiment outputs that could be needed for re-plotting
- if we later add plotting scripts, they should read from the saved `codex/experiments/.../results.json` files

Storage policy:

- keep compact tabular raw data only
- do not save full model checkpoints by default
- do not save full generated outputs / completions by default
- do not save tensor dumps unless required for a specific debugging task
- prune clearly failed or aborted run artifacts when they are no longer useful
- prefer one compact structured file per run over many verbose side artifacts
