# CDAA Shape Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IoU-gated, Surface_Crack-focused shape prior to CDAA so early low-quality predictions are not suppressed by unreliable aspect-ratio mismatch.

**Architecture:** Keep MDRH and BCRH unchanged. Extend `TaskAlignedAssigner` with one scalar gate parameter, pass it through `v8DetectionLoss`, and apply the gate only to the configured focus class. The gate uses the already-computed IoU tensor, so it adds no feature branch and only negligible elementwise computation.

**Tech Stack:** Python 3.11, PyTorch, pytest, Ultralytics YAML configuration.

## Global Constraints

- Keep `mdrh`, `bcrh`, and `cdaa` enabled in the stable experiment configuration.
- Do not add `cdaa_focus_max_topk`, `cdaa_focus_shape_weight`, or `bcrh_focal_gamma` to the active configuration.
- Keep `cdaa_shape_iou_gate` at `0.15` for the first controlled experiment.
- Preserve Pinhole and Delamination behavior by applying the new gate only to `cdaa_focus_class: 1`.
- Run tests with `D:\anaconda\envs\test\python.exe -m pytest`.

### Task 1: Add the failing CDAA gate tests

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: `TaskAlignedAssigner.shape_alignment` and the new `cdaa_shape_iou_gate` constructor argument.
- Produces: regression coverage for low-IoU protection and non-focus behavior.

- [ ] **Step 1: Add a test for low-IoU focus-class protection**

```python
def test_cdaa_shape_prior_is_gated_for_low_iou_focus_class():
    from ultralytics.utils.tal import TaskAlignedAssigner

    assigner = TaskAlignedAssigner(
        cdaa=True,
        num_classes=3,
        cdaa_shape_weight=0.2,
        cdaa_focus_class=1,
        cdaa_shape_iou_gate=0.15,
    )
    gt = torch.tensor([[[0.0, 0.0, 4.0, 1.0]]])
    pred = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    low_iou = torch.tensor([[[0.10]]])
    high_iou = torch.tensor([[[0.80]]])
    low = assigner.apply_shape_prior(torch.ones_like(low_iou), gt, pred, low_iou, torch.tensor([[[1]]]))
    high = assigner.apply_shape_prior(torch.ones_like(high_iou), gt, pred, high_iou, torch.tensor([[[1]]]))
    assert torch.allclose(low, torch.ones_like(low), atol=1e-6)
    assert high.item() < 1.0
```

- [ ] **Step 2: Add a test that non-focus classes retain the prior**

```python
def test_cdaa_shape_prior_remains_active_for_non_focus_class():
    from ultralytics.utils.tal import TaskAlignedAssigner

    assigner = TaskAlignedAssigner(
        cdaa=True,
        num_classes=3,
        cdaa_shape_weight=0.2,
        cdaa_focus_class=1,
        cdaa_shape_iou_gate=0.15,
    )
    gt = torch.tensor([[[0.0, 0.0, 4.0, 1.0]]])
    pred = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    metric = assigner.apply_shape_prior(
        torch.ones(1, 1, 1), gt, pred, torch.tensor([[[0.10]]]), torch.tensor([[[0]]])
    )
    assert metric.item() < 1.0
```

- [ ] **Step 3: Run only the new tests and verify they fail because the API is missing**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests/test_yolo11_innovation_modules.py -k "shape_prior_is_gated or shape_prior_remains_active" -q
```

Expected: FAIL with `TypeError` or `AttributeError` because `cdaa_shape_iou_gate` / `apply_shape_prior` does not yet exist.

### Task 2: Implement the minimal CDAA gate

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\utils\tal.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\utils\loss.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\cfg\default.yaml`

**Interfaces:**
- Consumes: `cdaa_shape_iou_gate: float` from model arguments.
- Produces: `TaskAlignedAssigner.apply_shape_prior(...)` and gated alignment metrics.

- [ ] **Step 1: Add the constructor parameter and helper in `tal.py`**

```python
cdaa_shape_iou_gate: float = 0.15,
```

Store it as `self.cdaa_shape_iou_gate = min(max(float(cdaa_shape_iou_gate), 0.0), 0.99)`.

Add:

```python
def apply_shape_prior(self, metric, gt_bboxes, pd_bboxes, overlaps, gt_labels):
    shape_quality = torch.ones_like(metric)
    valid = overlaps > 0
    if valid.any():
        shape_quality[valid] = self.shape_alignment(gt_bboxes.expand_as(pd_bboxes)[valid], pd_bboxes[valid])
    gate = ((overlaps - self.cdaa_shape_iou_gate) / (1.0 - self.cdaa_shape_iou_gate)).clamp(0.0, 1.0)
    focus = gt_labels.squeeze(-1).eq(self.cdaa_focus_class).unsqueeze(-1)
    effective_gate = torch.where(focus, gate, torch.ones_like(gate))
    return metric * shape_quality.pow(self.cdaa_shape_weight * effective_gate)
```

The implementation must preserve the existing tensor shapes used by `get_box_metrics`; use the already masked `gt_boxes` and `pd_boxes` tensors there if direct expansion would not match the repository's mask layout.

- [ ] **Step 2: Replace the direct shape multiplication in `get_box_metrics`**

Use the existing `overlaps` and class labels so the focus mask is aligned with `(batch, max_gt, anchors)`. The non-focus path must use the same exponent as before; the focus-class path uses the IoU gate.

- [ ] **Step 3: Pass the parameter from `v8DetectionLoss`**

Add:

```python
cdaa_shape_iou_gate=getattr(h, "cdaa_shape_iou_gate", 0.15),
```

next to the existing CDAA parameters.

- [ ] **Step 4: Register the explicit default**

Add this line next to `cdaa_shape_weight` in `ultralytics/cfg/default.yaml`:

```yaml
cdaa_shape_iou_gate: 0.15 # IoU gate before focus-class shape prior becomes active
```

### Task 3: Clean the stable experiment configuration

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\battery_defect_055_config.yaml`

- [ ] **Step 1: Add the explicit gate**

```yaml
cdaa_shape_iou_gate: 0.15
```

- [ ] **Step 2: Remove the stale unsupported option**

Remove:

```yaml
bcrh_focal_gamma: 0.0
```

### Task 4: Run the verification suite

**Files:**
- No additional production files.

- [ ] **Step 1: Run the new focused tests**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests/test_yolo11_innovation_modules.py -k "shape_prior_is_gated or shape_prior_remains_active" -q
```

Expected: both tests pass.

- [ ] **Step 2: Run the complete relevant tests**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests/test_yolo11_innovation_modules.py tests/test_battery_modules.py tests/test_validate_trained_model.py -q
```

Expected: exit code 0 and zero failures.

- [ ] **Step 3: Run syntax and configuration checks**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m compileall -q ultralytics train.py validate_trained_model.py
D:\anaconda\envs\test\python.exe -c "import yaml; yaml.safe_load(open('battery_defect_055_config.yaml', encoding='utf-8')); print('yaml ok')"
```

Expected: both commands exit 0 and print `yaml ok` for the second command.

- [ ] **Step 4: Report that accuracy still requires a remote controlled run**

Do not claim mAP improvement from unit tests. Compare the next remote run against the current `.552` aggregate and `.334` Surface_Crack mAP50-95 using the same seed, split, image size, batch, and validation script.
