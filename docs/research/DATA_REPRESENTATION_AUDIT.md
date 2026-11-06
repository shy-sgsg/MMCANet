# Data representation audit — Phase C1

**Date:** 2026-10-28  
**Scope:** authenticated `sen12_scene_dev1800_v2` train/development cohort only; 1,600 train and 200 development pairs. No test directory, image, or measurement was accessed.  
**Status:** source identity verified for this cohort; local numeric 2k subset remains unmapped.

## Source-backed representation

The official SEN1-2 supporting document describes SAR as 8-bit, single-channel images representing sigma-nought backscatter on a dB scale, and optical as 8-bit color images from bands 4/3/2. The official-source audit matched all 1,800 selected SAR/optical pairs to their source paths at decoded-pixel level. This verifies the selected files and their encoding description; it does not provide the byte-to-dB quantization table, source-product calibration metadata, or raw linear power values.

The current B1 v2 loader divides byte values by 255 and applies `log1p(20x)/log(21)` to SAR. Without a documented inverse from PNG DN to sigma-nought dB and then to linear power, this is an additional nonlinear transform of the encoded representation. It must not be described as converting SAR to calibrated intensity or as a physically correct log transform. Optical RGB is divided by 255 without per-image stretching; the channel-to-band assignment follows the official dataset description.

## Pixel profile of the verified cohort

The read-only profile reopened only the 1,800 frozen `train`/`val` composite PNGs from the sibling SOTANet cache. It checked the composite-file SHA-256 and both decoded panel hashes against the retained official-source verification record before updating counts. It read no test path. The resulting [`summary.json`](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/summary.json) reports 117,964,800 pixels per SAR channel and per optical channel.

| Channel | Mean DN | Population SD | DN=0 | DN=255 | 1st / 50th / 99th percentile |
|---|---:|---:|---:|---:|---|
| SAR | 143.756 | 53.846 | 0.673% | 2.609% | 10 / 146 / 255 |
| Optical R | 89.561 | 58.084 | 0.663% | 1.586% | 3 / 82 / 255 |
| Optical G | 91.649 | 54.078 | 0.682% | 1.609% | 4 / 84 / 255 |
| Optical B | 83.468 | 54.571 | 0.680% | 1.615% | 3 / 73 / 255 |

All 256 integer codes occurred in every channel. The endpoint counts are descriptive DN frequencies; the available transfer documentation does not establish whether DN=255 means sensor saturation, an encoding ceiling, or naturally mapped high values. The train and development subsets also have different aggregate SAR means (145.026 versus 133.592 DN); this is a cohort description, not an independent distribution test. Pixel counts are spatially correlated and must not be treated as independent sample counts.

## Evidence limits and next action

- These statistics belong to the official-source-verified 1,800-pair cohort, not the separate numeric-name `data/sar` / `data/opt` 2,000-pair subset.
- The model representation is a normalized transform of 8-bit source codes. No absolute radiometry, linear intensity, calibrated power, or physical ENL can be inferred from these values.
- Keep synthetic training conclusions in the representation domain. A real-SAR evaluation needs source product metadata, a defined physical-domain conversion, and a justified reference or independent measurement.
- Do not use these descriptive statistics to select a model or tune a threshold. No model was run and no gate changed.

## Sources

- Official SEN1-2 supporting document: [`SupportingDocument.txt`](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/SupportingDocument.txt).
- Official dataset paper: Schmitt, Hughes, and Zhu, [SEN1-2 dataset paper](https://doi.org/10.5194/isprs-annals-IV-1-141-2018).
- Source/pair verification: [`sen12_scene_dev1800_v2_auth_20261021_v1`](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md).
- Pixel-profile command and code: [`audit.py`](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/audit.py).
- Current preprocessing/degradation: [`degradation.py`](../../MMCANet/src/degradation.py), [`prepare_fixed_degradation.py`](../../scripts/prepare_fixed_degradation.py), and [`sen12_scene_dev1800_v2/b1_revalidation_v2.json`](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.json).

`test_set_accessed=false`, `test_set_used=false`, `training_or_inference_performed=false`, and `model_selection_performed=false` for this audit.
