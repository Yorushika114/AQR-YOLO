# Final Detection Ceiling Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three independently switchable training/model mechanisms that target late-stage mAP50-95 rather than only early optimization: multi-threshold candidate ranking, geometry-uncertainty prototype memory, and budgeted structured routing.

**Architecture:** Keep the standard YOLO11 detection branches and baseline path intact. Add MTCRL as a bounded auxiliary loss over top-K assigned candidates, GU-PMM as a bounded EMA prototype memory loss over detached high-quality positives and hard negatives, and BESR as a zero-initialized structured feature-routing residual in the detection head. Each mechanism is opt-in through `default.yaml`; disabled behavior must match the existing baseline path.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO11 detection loss/head/YAML, pytest.

## Global Constraints

- Preserve the existing baseline when `mtcrl=False`, `gupmm=False`, and `besr=False`.
- Do not remove or silently enable earlier SGH/CCSA/BDGC/PCRA/SRF/APCC experiments.
- Bound all auxiliary memory, candidate pairs, and routing groups to avoid quadratic image-wide work.
- Use zero-initialized residuals or disabled switches so loading `yolo11n.pt` remains valid.
- Validate wiring with CPU tests before any remote GPU training; a smoke test proves connectivity, not accuracy.

---

### Task 1: MTCRL candidate ranking loss

**Files:**
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_final_ceiling_modules.py`

**Interfaces:**
- Add `MultiThresholdCandidateRankingLoss(topk, max_pairs, warmup_epochs)` with `forward(pred_scores, target_scores, iou, uncertainty, valid) -> Tensor`.
- Construct it in `v8DetectionLoss` only when `mtcrl=True` and add its bounded scalar to the classification loss.
- Use a per-call epoch counter supplied by the criterion; the first `warmup_epochs` calls return a zero scalar without changing the baseline assignment.

- [x] **Step 1: Write failing tests** for pair ordering, zero output during warmup, finite gradients, and disabled baseline.
- [x] **Step 2: Run the focused tests and confirm failure because the class/config do not exist.**
- [x] **Step 3: Implement sparse same-target candidate pairs, IoU-lattice quality targets, DFL uncertainty weighting, and a softplus pairwise ranking loss with `max_pairs` truncation.
- [x] **Step 4: Wire only the opt-in flag and gains into `v8DetectionLoss`.
- [x] **Step 5: Run focused MTCRL tests and the existing battery tests.

### Task 2: GU-PMM prototype memory

**Files:**
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_final_ceiling_modules.py`

**Interfaces:**
- Add `GeometryUncertaintyPrototypeMemory(num_classes, feature_dim, prototypes_per_class, temperature, momentum)` with `forward(features, classes, geometry, quality, valid) -> Tensor`.
- Maintain buffers only; update prototypes under `torch.no_grad()` using reliable positives and hard negatives.
- Feed the detection head’s detached per-candidate embedding and geometry metadata to the criterion through an optional `prototype_features` output; with the flag disabled, no feature branch is created.

- [x] **Step 1: Write failing tests** for prototype initialization, EMA update only on valid samples, finite prototype loss, and disabled behavior.
- [x] **Step 2: Run the focused tests and confirm the expected missing-interface failure.
- [x] **Step 3: Implement normalized geometry-aware prototype distances and a bounded hard-negative contrastive objective.
- [x] **Step 4: Add an opt-in lightweight embedding projection to `StructuredDefectDetect` with zero-initialized output scale and connect it to the criterion.
- [x] **Step 5: Run prototype tests plus model build and one-step backprop tests.

### Task 3: BESR budgeted structured routing

**Files:**
- Modify: `ultralytics/nn/modules/battery.py`
- Modify: `ultralytics/nn/modules/head.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py` only if parser registration is required
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_final_ceiling_modules.py`

**Interfaces:**
- Add `BudgetedEdgeStructuredRouter(channels, groups, budget_ratio, node_dim)` with `forward(features) -> features` and a `last_gate_statistics` diagnostic.
- Route contiguous spatial groups using edge/contrast and feature uncertainty proxies; enforce a soft budget and graph smoothness regularizer returned by `regularization() -> Tensor`.
- `StructuredDefectDetect` applies BESR only when `besr=True`; its residual scale starts at zero and the normal Detect path is unchanged when disabled.

- [x] **Step 1: Write failing tests** for shape preservation, budget bound, zero-initialized identity, and gradient flow.
- [x] **Step 2: Run the focused tests and confirm the expected missing-class failure.
- [x] **Step 3: Implement grouped gates, budget penalty, contiguous-group smoothness, and a zero-initialized residual expert.
- [x] **Step 4: Wire the router into `StructuredDefectDetect` and expose only the necessary flags.
- [x] **Step 5: Run routing tests and YAML construction tests.

### Task 4: Joint integration and regression verification

**Files:**
- Modify: `train.py` only to add explicit experimental flags while preserving the current dataset/training values
- Modify: `ultralytics/cfg/models/11/yolo11n-sgh-ccsa-bdgc.yaml` only if the head arguments need the new opt-in defaults
- Test: `tests/test_final_ceiling_modules.py`

- [x] **Step 1: Add a test that builds the model with all three flags and completes one finite loss/backprop step.
- [x] **Step 2: Add model-build and disabled-switch coverage for the new head path.
- [x] **Step 3: Run `python -m compileall -q ultralytics train.py tests/test_final_ceiling_modules.py`.
- [x] **Step 4: Run `D:\anaconda\envs\test\python.exe -m pytest tests\test_final_ceiling_modules.py tests\test_sgh_ccsa_bdgc.py -q`.
- [x] **Step 5: Verify YAML construction, pretrained loading, and the connected detection-loss step; accuracy remains an experiment requiring remote training.

### Self-review checklist

- Every flag has an explicit disabled-path test.
- No module uses image-wide all-pairs comparisons.
- Prototype updates are detached and bounded, so they cannot corrupt gradients.
- BESR regularization is added only when enabled.
- The expected metric target is not guaranteed; final claims require equal-seed ablations and repeated runs.
