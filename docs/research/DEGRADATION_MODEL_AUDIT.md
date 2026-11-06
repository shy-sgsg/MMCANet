# Degradation model audit — Phase C1

**Date:** 2026-10-28  
**Scope:** canonical degradation code and frozen SEN1-2 B1 v2 configuration; source review plus synthetic unit tests only. No model forward pass, training run, test data, or new image evaluation.

## Frozen B1 v2 degradation chain

The run-bound configuration specifies the following order for each 256×256 normalized SAR pseudo-target:

1. Centered rectangular FFT low-pass with a 129×129 retained window (bins −64 through +64 on each axis).
2. No anisotropic blur.
3. Bilinear resize to 128×128 with `align_corners=false` (×2 SR input).
4. Clamp negative values to zero before the multiplier.
5. Multiply by an independent unit-mean Gamma factor with shape/rate `L=2`, directly in the normalized representation domain.
6. Do not upper-clamp or renormalize after the multiplier.

The canonical implementation is [`MMCANet/src/degradation.py`](../../MMCANet/src/degradation.py); the frozen parameters are in [`b1_revalidation_v2.json`](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.json). The 129-bin cutoff is the run-bound v2 setting; the CLI's legacy default of 64 is not interchangeable. Bilinear downsampling contributes additional passband attenuation, so the combined chain is not a sensor point-spread function.

## Statistical and physical interpretation

For a fixed nonnegative pre-speckle representation value `q`, the sampled `G ~ Gamma(L, rate=L)` has `E[G]=1` and `Var(G)=1/L`, so the implemented multiplier has `E[qG|q]=q` and `Var(qG|q)=q²/L`. At `L=2`, the conditional representation-space variance is `q²/2`. The official source describes the input codes as sigma-nought dB representations, and the B1 pipeline first normalizes/transforms these codes. Therefore `L=2` is a controlled synthetic stress parameter, not a measured number of sensor looks and not a calibrated intensity-speckle model.

The HR target is already a noisy 8-bit SAR representation. Applying a new multiplier after filtering and resizing composes synthetic multiplicative variation with residual source noise and nonlinear representation transforms. This is a controlled composite-noise assumption, not a claim about how a physical SAR sensor generated the pair. The separate derivation in [`representation_domain_gamma_composition_20261025.md`](representation_domain_gamma_composition_20261025.md) shows why applying the multiplier before versus after a log transform is not equivalent.

## Verification evidence

- Existing one-million-sample sampler audit passed mean/variance checks for `L=1,4,8` in declared intensity, representation, and amplitude modes; it validates sampler moments only, not the data or a sensor model.
- This continuation ran 38 synthetic unit tests covering canonical normalization, FFT cutoffs and real-valued behavior, Gamma moments/determinism, evaluator known cases, empty/degenerate masks, ratio direction, and fail-closed aggregation. All 38 passed. The selected tests used generated arrays and did not execute a model.
- The FFT support and bilinear response have prior synthetic regressions recorded in [`test_degradation_and_model.py`](../../tests/test_degradation_and_model.py) and [`frequency_resampler_response_20261025_v1`](../../experiments/data_audits/frequency_resampler_response_20261025_v1/report.md).

## C2 condition boundary

If C1 obtains new eligible groups, `L=8` (variance factor `1/8`) and `L=2` (`1/2`) can serve as candidate light/severe *representation-space* levels, with a fixed 2×/4× spatial pipeline and a single predeclared noise realization per identity/method comparison. They must not be called physical ENL or sensor looks. Before freezing C2, check the observed input range and model input contract, because no post-noise upper clamp allows values above 1.0. Do not run C2 on the already-used B1 v2 validation groups.

`test_set_accessed=false`, `test_set_used=false`, and `model_forward_or_training=false` for this audit. This document does not change B1's historical gate or authorize C2.
