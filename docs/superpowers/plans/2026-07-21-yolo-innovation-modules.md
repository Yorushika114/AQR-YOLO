# YOLO Innovation Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently switchable DIDC, MCRD, and CBSC modules to the local Ultralytics checkout without changing the default YOLO11 path.

**Architecture:** DIDC is a training-view and auxiliary-loss wrapper around the existing detector. MCRD is a shared candidate relation decoder called before the existing Hard-NMS. CBSC is a multi-view prediction utility that maps overlapping crop detections back to image coordinates and merges them before MCRD/NMS.

**Tech Stack:** Python 3.8+, PyTorch, existing Ultralytics tensor and box utilities, pytest.

## Global Constraints

- Default behavior remains unchanged when `didc`, `mcrd`, and `cbsc` are false.
- Reuse existing `v8DetectionLoss`, `non_max_suppression`, and box conversion utilities instead of duplicating them.
- Keep DIDC, MCRD, and CBSC independently configurable for ablation experiments.
- Preserve the original Hard-NMS path as the baseline.
- Use “拟实现/待验证” language for research effects; no accuracy claim is made by code completion.
- New Python files use the repository license header convention through the existing formatting workflow.

---

### Task 1: Add configuration and pure module contracts

**Files:**
- Modify: `ultralytics/cfg/default.yaml`
- Create: `ultralytics/data/didc.py`
- Create: `ultralytics/utils/mcrd.py`
- Create: `ultralytics/data/cbsc.py`
- Test: `tests/test_battery_modules.py`

**Interfaces:**
- `build_counterfactual_views(image, bboxes, cls, batch_idx, seed, erase_prob, insert_prob) -> dict`
- `mcrd_decode(candidates, image_shape, radius, topk, duplicate_threshold, independent_threshold, continuation_threshold) -> (Tensor, dict)`
- `generate_overlapping_crops(image_shape, crop_size, overlap, max_crops) -> list[CropView]`
- `map_boxes_to_crop(boxes, view) -> Tensor`, `map_boxes_to_global(boxes, view) -> Tensor`
- `fuse_view_candidates(candidates, source, match_iou) -> (Tensor, dict)`

- [ ] **Step 1: Write failing tests** for target count changes, crop round-trip mapping, duplicate merging, independent same-class retention, different-class retention, and disabled-path equality.
- [ ] **Step 2: Run `pytest tests/test_battery_modules.py -v`** and confirm failures are caused by missing module interfaces.
- [ ] **Step 3: Implement the smallest tensor-only helpers** with deterministic seeds, clipping, empty-input handling, and metrics dictionaries containing before/after counts.
- [ ] **Step 4: Run the focused test file** and confirm all pure-module tests pass.

### Task 2: Wire DIDC into detection training

**Files:**
- Modify: `ultralytics/models/yolo/detect/train.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `ultralytics/utils/loss.py` only if the existing criterion needs a shared helper for the auxiliary terms.
- Test: `tests/test_battery_modules.py`

**Interfaces:**
- Training preprocessing adds `img_erase`, `img_insert`, `bboxes_erase`, `cls_erase`, `batch_idx_erase`, `bboxes_insert`, `cls_insert`, and `batch_idx_insert` only when `args.didc` is true.
- The detection model computes the normal criterion for each view and combines them using `didc_weight`; the ordinary branch remains the exact default when disabled.

- [ ] **Step 1: Add a failing integration test** that calls `DetectionTrainer.preprocess_batch` with `didc=False` and `didc=True`, checking unchanged baseline keys and one fewer/one more target per edited image.
- [ ] **Step 2: Run the focused test** and confirm the enabled case fails before the wiring exists.
- [ ] **Step 3: Add preprocessing and shared-criterion dispatch** without copying `v8DetectionLoss`; retain original batch values and use separate branch dictionaries for labels.
- [ ] **Step 4: Add a small model-loss test** using a lightweight detection model or criterion stub to verify gradients and branch metrics are returned.
- [ ] **Step 5: Run focused DIDC tests and a one-batch CPU smoke test.**

### Task 3: Wire MCRD into prediction and validation

**Files:**
- Modify: `ultralytics/utils/nms.py`
- Modify: `ultralytics/models/yolo/detect/predict.py`
- Modify: `ultralytics/models/yolo/detect/val.py`
- Test: `tests/test_battery_modules.py`

**Interfaces:**
- `non_max_suppression(..., mcrd=False, mcrd_config=None)` preserves the current result exactly when false.
- When enabled, candidate counts are recorded as `nms_candidates`, `mcrd_candidates`, and `final_detections`.

- [ ] **Step 1: Add failing tests** for the false-path equality and the four MCRD geometric cases.
- [ ] **Step 2: Run them and confirm the enabled MCRD path fails.**
- [ ] **Step 3: Call the shared `mcrd_decode` before class-aware Hard-NMS**, preserving tensor device and dtype and leaving the original NMS implementation as the final suppression step.
- [ ] **Step 4: Pass the same options from predictor and validator.**
- [ ] **Step 5: Run focused tests and a CPU prediction smoke test with `mcrd=False`.**

### Task 4: Add CBSC multi-view prediction utilities and opt-in integration

**Files:**
- Modify: `ultralytics/models/yolo/detect/predict.py`
- Modify: `ultralytics/models/yolo/detect/val.py` only if validation can consume the same view-fusion contract without duplicating prediction logic.
- Test: `tests/test_battery_modules.py`

**Interfaces:**
- Crop metadata contains `offset`, `scale`, `orig_shape`, `crop_shape`, and `is_boundary`.
- The predictor accepts `cbsc=False` by default and uses `fuse_view_candidates` after mapping crop predictions to global coordinates.

- [ ] **Step 1: Add failing tests** for overlapping crop generation, boundary clipping, global-coordinate recovery, and duplicate crop detections.
- [ ] **Step 2: Run them and verify failures are feature failures.**
- [ ] **Step 3: Integrate crop generation and mapping through the existing predictor preprocessing path**, limiting crops with `cbsc_max_crops` and never recursing through the public predictor API.
- [ ] **Step 4: Feed fused candidates into the shared MCRD/NMS path when enabled.**
- [ ] **Step 5: Run focused tests and compare a disabled CBSC prediction against the unmodified baseline path.**

### Task 5: Regression verification and research handoff

**Files:**
- Modify: `ultralytics/cfg/default.yaml` if config validation reveals a naming collision.
- Modify: `tests/test_battery_modules.py` for any regression discovered by the focused suite.

- [ ] **Step 1: Run `pytest tests/test_battery_modules.py -v`.**
- [ ] **Step 2: Run existing lightweight tests covering Python API and detection prediction.**
- [ ] **Step 3: Run `ruff check` on changed Python files if Ruff is installed.**
- [ ] **Step 4: Run a CPU one-batch smoke test for baseline, `+DIDC`, `+MCRD`, `+CBSC`, and the combined switches where dependencies permit.**
- [ ] **Step 5: Report changed files, test evidence, known limitations, and the required future ablation metrics without claiming unmeasured gains.**

## Scope Review

The implementation covers the local code requirements that can be validated without a trained relation-label dataset: independent switches, shared loss reuse, DIDC auxiliary constraints, pre-NMS MCRD placement, optional trainable relation decoding, CBSC coordinate mapping, CBSC training consistency, prediction/validation fusion, baseline preservation, and focused tests. Accuracy, latency, FPS, memory, and relation-decoder training results remain experimental measurements.

## Implementation Status (2026-07-21)

- Completed: DIDC deterministic erase/insert views, branch detection losses, outside-region stability loss, target-response direction loss, metadata, and CPU backward smoke test.
- Completed: MCRD geometric relation decoding, duplicate/continuation grouping, independent/cross-class retention, optional `MCRDRelationDecoder`, supervised relation loss API, and shared NMS integration.
- Completed: CBSC crop metadata and label projection, global coordinate mapping, training crop loss, differentiable cross-view response/class consistency, shared prediction/validation fusion, and final MCRD/NMS integration.
- Completed: default-off configuration, focused tests, baseline tests, real CPU prediction smoke tests for baseline/MCRD/CBSC, and one-batch DIDC/CBSC backward tests.
- Pending experiment: dataset audit, fixed-budget baseline and ablations, defect-level metrics, latency/FPS/memory measurements, and training a relation decoder on relation labels.
