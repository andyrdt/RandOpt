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
8. Phase 1 is now complete for:
   - `Qwen/Qwen2.5-0.5B-Instruct`
   - `Qwen/Qwen2.5-1.5B-Instruct`
   - `Qwen/Qwen2.5-3B-Instruct`
   - `Qwen/Qwen2.5-7B-Instruct`
   under both:
   - released sampler (`per_parameter`)
   - fixed RNG sampler (`global_stream`)
9. Compact analysis tables and first-pass figures have been regenerated from saved raw results.

Already learned:

- The RNG bug changes the perturbation reward distribution materially.
- The bug does not obviously destroy `RandOpt`; on the first slice, top-1 got worse but top-4 ensemble improved.
- Sigma matters under the fixed RNG sampler, and very large sigma is destructive.
- The Phase 1 cross-scale picture is mixed:
  - `0.5B`: both samplers are weak
  - `1.5B`: released sampler is clearly better at `K=1` and `K=2`, with similar `K=4`
  - `3B`: released sampler is better at all tested `K`
  - `7B`: fixed RNG slightly improves `K=1` and `K=2`, while released remains slightly better at `K=4`
- The current density proxy (`hit-rate > base train`) does not increase monotonically through `7B`, so Phase 2 needs to disentangle:
  - threshold effects from a strong base model
  - sigma calibration effects
  - genuine deviations from a simple scaling story

Important caveat:

- Phase 1 used `N=40`, so all cross-scale conclusions should be treated as preliminary.
- The Phase 1 results are strong enough to prioritize follow-up experiments, but not strong enough to support confident scientific claims on their own.
- In particular, `5`-point swings in `K=4` accuracy at these sample sizes should not be over-interpreted.
- The `7B` hit-rate drop should be treated as suggestive evidence for the fixed-sigma critique, not as a final verdict.

Execution status:

- Phase 2 is complete.
  - the paper-split log-spaced density sweep completed for:
    - `0.5B`
    - `1.5B`
    - `3B`
    - `7B`
  - sigma grid:
    - `1e-5`
    - `3e-5`
    - `1e-4`
    - `3e-4`
    - `1e-3`
    - `3e-3`
    - `1e-2`
  - the main density and `K=1` plots have been regenerated from those completed runs
- Phase 3 is complete for the core model set.
  - relative-noise runs completed for:
    - `0.5B`
    - `1.5B`
    - `3B`
    - `7B`
  - weight-scale diagnostics have also been completed for:
    - `32B`
  - the refreshed Phase 3 comparison figures now include the completed `7B` tensor-std sweep
- Opportunistic extension:
  - if a genuinely free extra GPU appears without displacing another user's work, use it for the already-noted `0.5B` small-sigma Phase 2 extension
  - current intended `0.5B` extension grid:
    - `0.000125`
    - `0.00025`
    - `0.0005`
    - `0.001`
    - `0.002`
    - `0.003`
    - `0.005`
  - current state:
    - launched on GPU `5`
    - completed `0.000125` with no ensemble gain
    - completed `0.00025` with no ensemble gain
    - completed `0.0005` with no ensemble gain
    - active cell is now `0.001`
- Queued next-priority batch after current in-flight jobs finish:
  - run a log-spaced fixed-RNG absolute-noise density sweep focused on hit-rate
  - purpose:
    - identify the sigma that maximizes hit-rate for each model size
    - test whether peak hit-rate increases monotonically with model size under the corrected sampler
  - configuration:
    - sampler: `global_stream`
    - scale mode: absolute
    - models:
      - `Qwen/Qwen2.5-0.5B-Instruct`
      - `Qwen/Qwen2.5-1.5B-Instruct`
      - `Qwen/Qwen2.5-3B-Instruct`
      - `Qwen/Qwen2.5-7B-Instruct`
    - sigma grid:
      - `0.00001`
      - `0.00003`
      - `0.0001`
      - `0.0003`
      - `0.001`
      - `0.003`
      - `0.01`
    - dataset split:
      - `train_samples=200`
      - `test_samples=2000`
      - this matches the paper's stated protocol on `Countdown`
    - population size:
      - `N=100` per `(model, sigma)` pair
    - primary output:
      - hit-rate above base-train accuracy
    - practical note:
      - keep top-`K` minimal for this batch
      - with `population_size=100`, `top_k_ratios=0.01` gives `K=1`
      - the main output remains hit-rate, not ensemble accuracy
  - execution refinement:
    - let the currently active cells finish cleanly
    - do not keep spending GPUs on lower-priority loop tails once those active cells are done
    - use the next free-GPU launches for this density sweep instead
  - operationalization:
    - handoff watchers are installed so each currently busy GPU should roll directly from its active cell into the density sweep for the matching model once that cell exits

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
- determine whether the `7B` hit-rate drop is primarily a sigma-calibration issue
- increase statistical power relative to Phase 1 before drawing stronger conclusions

Configuration:

- fixed RNG only: `global_stream`
- task: `countdown`
- use the same `100/100` split for comparability
- focus on the models where Phase 1 is actually ambiguous or scientifically interesting:
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
- do not prioritize `0.5B` here unless needed for a specific sanity check
- use separate per-sigma runs rather than mixed-sigma populations for this phase
- prefer higher-`N` runs over broader but underpowered sweeps
- target population size:
  - default: `N=200`
  - if runtime is clearly prohibitive on `7B`, allow a smaller fallback there, but document it explicitly
- sigma sweep:
  - `0.0005`
  - `0.001`
  - `0.002`
  - `0.003`
  - `0.005`

Follow-up extension to keep explicitly on deck:

- if the current Phase 2 curves keep suggesting that smaller models only become productive at the very bottom of the absolute-noise grid, run a smaller-sigma extension
- prioritize:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
- first additional sigma values:
  - `0.000125`
  - `0.00025`
- use this both as:
  - a Phase 2 extension
  - a possible revisit of the Phase 1 interpretation for the smallest model sizes

Explicit expectation:

- yes, `0.5B` should eventually be included in this follow-up
- but not by interrupting the current higher-value `1.5B / 3B / 7B` Phase 2 batch mid-flight
- the small-model extension is meant to answer whether the smallest scales only look weak because the absolute-noise grid is too coarse or too large

Primary measurements:

1. Mean sampled train reward vs sigma
2. Hit-rate vs sigma
3. Best ensemble accuracy vs sigma
4. Reward delta vs base-train accuracy

Important note:

- this does not fully resolve the critique's cross-scale calibration issue
- it is still a useful practical check for whether the released sigma choices are unstable under the corrected sampler
- specific decision goal:
  - determine whether the odd `7B` density proxy is explained by sigma choice rather than by the RNG fix itself
- interpretation note:
  - connect any `7B` fixed-sigma failure explicitly to the critique's Issue 2 rather than treating it as a generic anomaly

### Phase 3: Relative-norm perturbation ablation

Purpose:

- test whether the paper's conclusions depend strongly on using constant absolute additive noise
- check whether a scale-aware perturbation changes the observed density or `RandOpt` behavior
- directly inspect whether parameter scales are comparable across:
  - layers within a model
  - model sizes within the Qwen family

Configuration:

- use the corrected sampler (`global_stream`) as the baseline for comparison
- compare two perturbation families:
  - absolute additive noise
  - per-tensor scale-aware additive noise
- task: `countdown`
- use the same `100/100` split for comparability
- focus on representative model sizes rather than rerunning every point if runtime is tight:
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
- compare against the already-completed Phase 1 / Phase 2 absolute-noise baselines instead of rerunning them unnecessarily
- do not begin this phase until all perturbation helper paths are consistent with the selected perturbation mode
  - specifically fix:
    - `apply_perturbation`
    - `apply_averaged_perturbations`
    - any other helper that still hardcodes per-parameter seeding or absolute scaling

Additional diagnostic for this phase:

- compute weight-scale statistics on the base pretrained models before perturbation
- primary statistics:
  - per-tensor standard deviation
  - per-tensor RMS
  - optionally mean absolute value
- aggregate these by:
  - layer index
  - exact named-parameter family wherever possible, not just coarse components
    - `q_proj.weight`
    - `q_proj.bias`
    - `k_proj.weight`
    - `k_proj.bias`
    - `v_proj.weight`
    - `v_proj.bias`
    - `o_proj.weight`
    - `gate_proj.weight`
    - `up_proj.weight`
    - `down_proj.weight`
    - `input_layernorm.weight`
    - `post_attention_layernorm.weight`
    - `embed_tokens.weight`
    - `lm_head.weight`
    - final `norm.weight`
- keep weights and biases separate in the summaries and plots
- focus on:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen/Qwen2.5-32B-Instruct`

Diagnostic goal:

- determine whether absolute additive `sigma` is even acting on comparable parameter scales across:
  - layers
  - model sizes
- determine which exact parameter families dominate the raw scale differences
  - especially biases vs weights
  - and attention subfamilies vs MLP / norm / embedding families
- give direct visibility into the parameter-scale critique instead of only testing it through the relative-norm ablation

### Phase 4: Optional 32B replication

Purpose:

- check whether the trends seen on `0.5B` through `7B` continue at a substantially larger scale
- provide an optional bridge toward the larger-model part of the paper without making `32B` the place where protocol choices are debugged

Priority:

- optional and later
- do not begin until the core Phase 2 and Phase 3 conclusions are stable enough that we know what perturbation protocol we actually want to test

Likely configuration:

- model:
  - `Qwen/Qwen2.5-32B-Instruct`
- task:
  - `countdown`
- perturbation family:
  - whichever protocol survives Phases 2 and 3 as the most informative baseline
- likely runtime note:
  - expect multi-GPU tensor parallelism to be required
  - do not assume 1-GPU execution will be practical even on 80GB cards

Goal:

- treat `32B` as a scale-extension check, not as part of the method-debugging loop

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

### Figure 6: Weight Scale by Layer

Question:

- are raw weight scales comparable across layers and across model sizes?
- which exact named parameter families are scale outliers under the paper's "perturb all named params equally" setup?

Plot:

- x-axis: layer index
- y-axis: weight scale statistic
  - primary: tensor std or RMS
- one panel per model size
- separate series for exact parameter families where readable
  - especially `q/k/v/o` weights and biases
  - MLP weights
  - layernorm weights

Companion model-scale views:

- model-size vs median tensor std for:
  - attention families
  - MLP / norm families
  - shared tensors such as embeddings, final norm, and lm head

Companion summary:

- whole-model table with:
  - median tensor std
  - min / max tensor std
  - spread across layers

## Execution Order

1. Treat Phase 1 as complete and use it as the locked baseline.
2. Before running Phase 2 or Phase 3, fix the remaining perturbation helper paths so perturbation behavior is consistent everywhere it may be used.
3. Run targeted Phase 2 fixed-RNG sigma sweeps on:
   - `1.5B`
   - `3B`
   - `7B`
4. Refresh sigma-sensitive summary tables and build the updated Figure 3.
5. Run the weight-scale diagnostic on the base models and build Figure 6.
6. Implement the Phase 3 relative-norm perturbation mode using per-tensor base-weight standard deviation.
7. Run the Phase 3 relative-norm sweeps on:
   - `1.5B`
   - `3B`
   - `7B`
8. Build Figure 5 and compare against the existing absolute-noise baselines.
9. Update the final summary and conclusions once both phases are complete.

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
- This is important: keep working autonomously until the current experiment plan is completed or a real blocker is reached.
- Do not stop early just because runs are long or inconvenient.
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
