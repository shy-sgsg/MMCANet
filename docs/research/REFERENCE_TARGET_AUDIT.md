# Reference target audit — Phase C1

**Date:** 2026-10-28  
**Status:** target semantics identified for the verified SEN1-2 cohort; no independent clean SAR reference is available in the current protocol.

## What the target is

The B1 v2 high-resolution target is the same source SAR patch loaded from the official 8-bit sigma-nought-dB representation, converted to `[0,1]` by byte scaling and passed through the configured log normalization. It is therefore a **noisy SAR pseudo-target** in a normalized representation space. It is not a clean image of latent reflectivity and is not a calibrated linear-intensity ground truth.

For each training pair, the low-resolution SAR input is deterministically derived from that same target using the frozen FFT/resize/negative-clamp chain, followed by representation-domain Gamma multiplication. Thus the model is evaluated for agreement with a noisy target that is statistically related to its input. PSNR, SSIM, and the additional structure/scatterer metrics describe this proxy-target agreement under the declared normalization and masks; they do not directly measure clean-SAR recovery.

## What is not available

The selected 1,800-pair cohort does not include raw Sentinel-1 SAFE/SLC data, a documented DN↔sigma-nought transfer table, independent multilook estimates, or repeated same-scene acquisitions that could establish a clean reference. The official source folder match proves file provenance for this cohort, not a noise-free target or exact acquisition-time pairing. The separate project numeric 2,000-pair subset remains unmapped and has no verified target lineage.

## Consequences

- Describe current results as reconstruction against normalized, noisy 8-bit pseudo-targets under controlled synthetic degradation.
- Keep PSNR/SSIM and target-derived TBR/edge/bright-region metrics as representation-domain diagnostics. Pair them with structure and ratio measures, but do not label them calibrated radiometric fidelity or real despeckling quality.
- Track A (controlled synthetic degradation) is feasible on the verified source cohort only after a new eligible source-group protocol is available.
- Track B (real SAR evaluation) is not currently executable as defined. It needs real low-resolution observations and a defensible independent reference or measurement protocol; a second synthetic transform of the existing pseudo-target is not Track B.
- Any future clean-reference claim requires a documented target-generation process and an explicit argument about residual target noise and its relation to the LR input.

## Sources

- Run-bound semantics and parameters: [`b1_revalidation_v2.json`](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.json).
- Fixed degradation and same-target derivation: [`degradation.py`](../../MMCANet/src/degradation.py), [`prepare_fixed_degradation.py`](../../scripts/prepare_fixed_degradation.py), and [`MMCANetV2/data.py`](../../MMCANetV2/data.py).
- Official SEN1-2 encoding description: [`SupportingDocument.txt`](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/SupportingDocument.txt).
- Source and split constraints: [`sen12_scene_dev1800_v2_auth_20261021_v1`](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md) and [`sen1_2_train_val_spatial_overlap_audit.md`](sen1_2_train_val_spatial_overlap_audit.md).

No test data was accessed, no model was run, and no historical result or gate was altered.
