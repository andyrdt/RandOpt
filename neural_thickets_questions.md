# Potential Issues in "Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights"

**Paper:** Gan & Isola, arXiv:2603.12228v1, March 2026
**Code:** https://github.com/sunrainyg/RandOpt

---

## Overview

"Neural Thickets" claims that the neighborhood around pretrained weights is dense with diverse task-specific experts, that this density scales with model size, and that a simple random-guessing-plus-ensembling algorithm (RandOpt) can exploit this structure competitively with gradient-based post-training methods. The underlying intuition — that pretraining creates weight-space regions rich with useful nearby solutions — is plausible and directionally supported by adjacent work on intrinsic dimensionality, LoRA, and low-rank fine-tuning curvature. However, examination of the released codebase reveals that the experiments do not test the paper's central mathematical claim as stated, and a secondary confound undermines the scaling-law analysis.

---

## Issue 1 (Fatal): The Implementation Does Not Sample the Distribution the Paper Claims to Study

### The claim

The paper defines "solution density" (Eq. 1) as the probability that a random perturbation $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ improves the base model. The perturbation model (Eq. 3) is $\theta' = \theta + \sigma \cdot \epsilon(s)$, where $\epsilon$ is drawn from an isotropic Gaussian over the full parameter space. The entire theoretical narrative — "neural thickets," the scaling law of density, the diversity analysis, Spectral Discordance — rests on perturbations being **independent isotropic Gaussian noise** applied to every parameter.

### What the code actually does

In `utils/worker_extn.py`, the perturbation routine is:

```python
def perturb_self_weights(self, seed, noise_scale, negate=False):
    self._set_seed(seed)
    scale = float(noise_scale)
    sign = -1.0 if negate else 1.0
    for name, p in self.model_runner.model.named_parameters():
        gen = torch.Generator(device=p.device)
        gen.manual_seed(int(seed))                # ← SAME seed every iteration
        noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
        if self._should_perturb(name):
            p.data.add_(sign * scale * noise)
        del noise
```

The generator is **re-created and re-seeded to the same value** for every parameter tensor. This means:

- **Every parameter tensor of the same shape receives byte-for-byte identical noise.** In a transformer, `q_proj` and `o_proj` weight matrices across *all* layers receive the exact same perturbation vector when their shapes match.
- **Tensors of different shapes share a common random prefix.** A (4096, 4096) tensor and a (4096, 1024) tensor will have their first 4,194,304 noise values be identical, since both generators start from the same state.

### Quantifying the impact

For a Qwen2.5-3B-scale model (~324 parameter tensors), there are only **5 unique shapes**. Each shape group contains up to 72 tensors that all receive identical noise:

| Shape | Tensors receiving identical noise | Elements per tensor |
|-------|----------------------------------|-------------------|
| (2048, 2048) | 72 (q\_proj, o\_proj across layers) | 4,194,304 |
| (512, 2048) | 72 (k\_proj, v\_proj across layers) | 1,048,576 |
| (5504, 2048) | 72 (gate\_proj, up\_proj across layers) | 11,272,192 |
| (2048, 5504) | 36 (down\_proj across layers) | 11,272,192 |
| (2048,) | 72 (layer norms across layers) | 2,048 |

| | Claimed | Actual |
|---|---|---|
| **Perturbation dimensions** | ~1.6 billion | ~28 million |
| **Dimensionality ratio** | 1.0 | 0.017 |

The code searches a **57× lower-dimensional** subspace than the paper's math describes — one where the perturbation is effectively a single "layer-edit template" stamped identically across the entire network.

### Why this is the most fundamental problem

The paper's headline claim is that task experts are *dense in an isotropic Gaussian neighborhood* around pretrained weights. The experiments never sample from that neighborhood. They sample from a tiny, highly structured subspace. The "thickets" phenomenon observed in Figures 2–4 may be a property of this particular tied-noise family — which effectively searches over layer-uniform model edits — rather than a property of the full isotropic neighborhood. These are qualitatively different claims.

This also has implications for *why RandOpt works in practice*. Stamping the same perturbation template across every layer is a coherent architectural edit, closer in spirit to a rank-1 LoRA-like modification than to random noise in billion-dimensional space. This is a much friendlier search space. If the RNG is fixed so that perturbations are truly independent across parameters, RandOpt's practical performance may also degrade — not just the density measurements. (See Experiment 3 below.)

### A notable irony

The `utils/repro_seed.py` script — intended for reproducing perturbed models outside the vLLM runtime — uses a single `torch.Generator()` seeded once, advancing naturally across all parameters. This is the *correct* isotropic Gaussian implementation that the paper describes. It's the main experimental code path in `worker_extn.py` that doesn't match. Models "reproduced" with the utility script would not match the models actually evaluated during experiments.

---

## Issue 2 (Major Confound): Fixed σ Across Model Scales Confounds the Density Scaling Law

### The claim

Figure 3 shows "solution density increases monotonically with model scale," using Qwen-2.5 models from 0.5B to 32B. The paper explicitly states (Section 2.1) that all experiments use $\sigma = 0.005$.

### The problem

The same per-parameter σ has a vastly different functional impact at different model scales. The paper's own Figure 12 shows this directly:

| Model | GSM8K: % perturbations ≥ base | Countdown: % perturbations ≥ base |
|-------|-------------------------------|-----------------------------------|
| 0.5B  | 0%  | 8%  |
| 1.5B  | 18% | 48% |
| 3B    | 37% | 54% |
| 32B   | 64% | 60% |

For the 0.5B model, the perturbation is catastrophically disruptive (centering around ~85% of base accuracy on GSM8K). For the 32B model, it barely moves the needle (centering around ~100%). This is expected: larger models have more redundancy, and each parameter contributes less to any given output.

The deeper issue is that the paper's neighborhood is defined in **raw weight coordinates**, which are not comparable across model sizes or parameterizations. A fixed σ in parameter space is an arbitrary choice unless calibrated to something function-space-based — equal KL divergence, equal logit perturbation, or equal average performance drop. The paper itself quietly acknowledges σ sensitivity: in the appendix's 1D-signal experiments, different pretraining regimes require different σ values "large enough to show functional variation."

### What this means for the scaling law

The "density scales with model size" finding can be partly reinterpreted as "robustness to a fixed raw-coordinate perturbation scales with model size." That said, this is a confound rather than a clean disproof. Figure 3 shows density increasing even at thresholds *above* base performance (103%, 105%, and up to 115% on Countdown), so the effect is not purely that large-model perturbations are no-ops. Resolving this definitively requires the calibrated-σ control experiment described in Experiment 2 below.

---

## Issue 3 (Moderate): ROCStories Evaluation Bug Inflates Accuracy

In `data_handlers/rocstories.py`, the methods `is_answer_correct` and `is_voted_answer_correct` return a **float** between 0.0 and 1.0 representing partial credit (60% position accuracy + 40% adjacent-pair bonus):

```python
def is_answer_correct(self, response: str, ground_truth) -> float:
    ...
    return self._compute_lenient_accuracy(pred_labels, gold_labels)
```

But the ensemble evaluator in `randopt.py` (line 252) tests correctness with a bare truthiness check:

```python
is_correct = handler.is_voted_answer_correct(final, data["ground_truth"])
if is_correct:
    correct += 1
```

In Python, any nonzero float is truthy. A prediction that gets 2/5 positions correct (score ≈ 0.4) counts as **fully correct**. This inflates the ROCStories accuracy numbers in Table 4. This bug does not affect the other benchmarks, which return proper booleans or use separate correctness-checking logic.

---

## How the Issues Compound

Issues 1 and 2 interact and both push in the direction of making the density scaling law look stronger than it may actually be.

The **structured perturbation** (Issue 1) searches a much friendlier subspace than true isotropic noise. Layer-uniform edits are coherent model modifications, not random destruction — they're more likely to produce functional specialists than independent noise would be. This inflates the measured density relative to what the paper's math predicts.

The **uncalibrated σ** (Issue 2) makes large models appear to have higher density partly because the perturbation barely affects them. A perturbation that doesn't move the model much will, by definition, rarely degrade it — inflating the count of "task-improving" perturbations.

Together, these confounds mean the quantitative density and diversity measurements (Figures 2, 3, 4, 7, 12) cannot be taken at face value as evidence for the paper's stated claims.

---

## Proposed Validation Experiments

### Experiment 1: Fix the RNG and re-measure solution density

**Goal:** Determine whether the "thickets" phenomenon survives under true isotropic Gaussian perturbations.

**Method:**
- Modify `perturb_self_weights` to use a single generator seeded once before the parameter loop (delete the per-tensor `gen = torch.Generator(); gen.manual_seed(int(seed))` and seed once outside the loop — essentially matching the logic already in `repro_seed.py`).
- Re-run the solution density measurement (Figure 3) across Qwen2.5 0.5B → 32B.
- Compare density curves under the original (correlated) and corrected (independent) perturbation schemes.

**What this resolves:** If thickets persist under true isotropic noise, the paper's core claim is rescued despite the implementation bug. If density collapses, the phenomenon was an artifact of the structured perturbation family.

### Experiment 2: Calibrate σ per model size

**Goal:** Disentangle "density of task experts" from "robustness to noise."

**Method:**
- For each model size, sweep σ and measure the average relative accuracy change.
- Choose σ per model such that the mean relative accuracy degradation is constant (e.g., 10% relative drop) across all scales.
- Re-measure solution density at these calibrated σ values.
- Alternative calibration targets: matched KL divergence between base and perturbed output distributions, or matched average logit perturbation magnitude.

**What this resolves:** If the density scaling law holds under functionally-normalized perturbations, it reflects genuine geometric structure. If it flattens or reverses, the original scaling law was a perturbation-sensitivity artifact.

### Experiment 3: Ablate structured vs. independent noise on RandOpt performance

**Goal:** Determine whether RandOpt's competitive performance depends on the (accidental) structure in the perturbations.

**Method:**
- Run RandOpt with the corrected (independent) perturbation scheme on Countdown and GSM8K with Qwen2.5-3B-Instruct and OLMo3-7B-Instruct.
- Use the same N, K, and σ settings as the paper (N=5000, K=50).
- Compare accuracy against the original (correlated) results from Table 4.

**Hypothesis:** The layer-uniform structure of the correlated perturbations may be *why* RandOpt works — effectively a coherent model edit rather than random noise. Under independent noise, the same σ may be more destructive, requiring either re-tuning σ or accepting lower performance. If RandOpt's results degrade substantially, this reframes the contribution: the interesting finding would be that *structured, layer-uniform perturbations* are effective for post-training, not that *random Gaussian perturbations* are.

### Experiment 4: Confirm and quantify the ROCStories bug

**Goal:** Obtain corrected ROCStories numbers.

**Method:**
- Fix `is_voted_answer_correct` to return `True` only when score == 1.0 (exact match on all 5 positions), or fix the ensemble evaluator to threshold explicitly.
- Re-run ROCStories evaluation for at least Qwen2.5-3B-Instruct and compare against Table 4.

### Experiment 5: Verify Spectral Discordance under corrected noise

**Goal:** Test whether the "specialists, not generalists" finding (Figure 3b, Figure 4) holds under independent perturbations.

**Method:**
- With the corrected RNG, sample 500 perturbations and evaluate across all 7 tasks.
- Recompute Spectral Discordance $\mathcal{D}$ and the PCA clustering analysis.

**Rationale:** Under the correlated scheme, all layers receive the same noise template. This may artificially create coherent "modes" of behavior — a perturbation that consistently strengthens or weakens a specific computational motif across every layer. Under independent noise, the per-layer effects would be less coherent, and the diversity patterns could differ substantially.

---

## Summary Table

| Issue | Severity | Nature | Impact on paper claims |
|-------|----------|--------|----------------------|
| Correlated noise (not isotropic Gaussian) | **Fatal** | Implementation ≠ math | Central density/diversity measurements are conducted under a different distribution than defined; headline claim is unsupported as stated |
| Fixed σ across model scales | **Major confound** | Experimental design | Scaling law of density is confounded with robustness to noise; requires a missing control experiment to resolve |
| ROCStories scoring bug | **Moderate** | Evaluation bug | One benchmark's numbers are inflated in Table 4 |
| Reproduction utility mismatch | **Minor** | Practical issue | Third-party reproduction would yield different models than those evaluated |

### What may survive

The paper's *intuition* — that pretraining creates weight-space regions rich with nearby task-specific solutions — is likely directionally correct. LoRA's success, intrinsic dimensionality results (Aghajanyan et al., 2020), the low-rank curvature findings of Liang et al. (2026), and the effectiveness of evolutionary strategies for LLM post-training (Qiu et al., 2025) all point in this direction. It is entirely possible that the thickets phenomenon is real and that fixing the implementation would confirm it. But until the proposed experiments are run, the specific quantitative claims — the scaling law, the diversity measurements, and RandOpt's competitiveness under the stated perturbation model — rest on evidence that does not match the paper's stated methodology.
