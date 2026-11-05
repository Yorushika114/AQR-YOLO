# TCAA and ACED Quality-Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the recall-versus-localization regression caused by TCAA adding unreliable positives that ACED then overweights.

**Architecture:** Keep the names TCAA and ACED. TCAA will allocate extra positives only when a target is both unresolved and currently learnable; ACED will focus on refinable medium/high-IoU positives instead of amplifying nearly zero-IoU assignments.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO, pytest.

## Global Constraints

- Keep the public names `TCAA` and `ACED`.
- Do not hard-code class names or indices.
- Do not add inference-time computation.
- Preserve the current YOLO loss API.
- Work directly in the local non-Git checkout.

---

### Task 1: Quality-gated TCAA

**Files:**
- Modify: `tests/test_tcaa_aced_dfl.py`
- Modify: `ultralytics/utils/tal.py`
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`

**Interfaces:**
- Consumes: detached best candidate quality per GT.
- Produces: TCAA dynamic top-k bounded by learnable residual and reliable candidate count.

- [ ] **Step 1: Add the failing unreliable-target test**

```python
def test_tcaa_does_not_expand_an_unreliable_zero_quality_target():
    assigner = TopologyConflictAwareAssigner(topk=1, topk_extra=3, quality_floor=0.25)
    mask_pos, _, _ = assigner.get_pos_mask(zero_scores, disjoint_boxes, labels, targets, anchors, valid)
    assert mask_pos.sum() == 1
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests\test_tcaa_aced_dfl.py::test_tcaa_does_not_expand_an_unreliable_zero_quality_target -q
```

Expected: fail because `quality_floor` is not accepted.

- [ ] **Step 3: Implement learnable residual gating**

```python
learnable_hardness = 4.0 * best_quality * (1.0 - best_quality)
reliable = (candidate_quality >= quality_floor) & (overlaps >= min_iou)
max_reliable_topk = reliable.sum(-1).clamp_min(topk)
dynamic_topk = minimum(topk + round(topk_extra * strength), max_reliable_topk)
```

- [ ] **Step 4: Add defaults and run TCAA tests**

Add `tcaa_quality_floor: 0.25` and `tcaa_min_iou: 0.20`, then run the focused test file.

### Task 2: Refinable-band ACED

**Files:**
- Modify: `tests/test_tcaa_aced_dfl.py`
- Modify: `ultralytics/nn/modules/battery.py`
- Modify: `ultralytics/cfg/default.yaml`

**Interfaces:**
- Consumes: aligned positive-box IoU.
- Produces: ACED weights highest for refinable positives and low for both unusable and converged positives.

- [ ] **Step 1: Add the failing localization-weight test**

```python
def test_aced_weights_refinable_localization_above_zero_iou_noise():
    weights = module._localization_weights(torch.tensor([0.0, 0.5, 1.0]))
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
```

- [ ] **Step 2: Verify RED**

Expected: fail because `_localization_weights` does not exist.

- [ ] **Step 3: Replace monotonic hard weighting**

```python
refinability = 4.0 * iou * (1.0 - iou)
weights = hardness_floor + refinability.pow(hardness_power)
```

Set `aced_hardness_power: 1.0` and `aced_variance_gain: 0.1`.

- [ ] **Step 4: Run complete verification**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests\test_tcaa_aced_dfl.py -q
D:\anaconda\envs\test\python.exe -m compileall -q ultralytics train.py
```

Expected: all tests pass and compilation exits with code 0.
