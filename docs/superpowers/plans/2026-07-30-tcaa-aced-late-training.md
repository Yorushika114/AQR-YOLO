# TCAA and ACED Late-Training Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TCAA and ACED continue focusing optimization on unresolved Surface_Crack matching and Delamination localization errors during long training.

**Architecture:** Keep the public names `TCAA` and `ACED`. Replace TCAA's geometry-only positive-count rule with a geometry-plus-current-residual rule, and replace ACED's target-score weighting with localization-hardness weighting plus DFL variance calibration. Both remain train-only and category agnostic.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO detection loss, pytest.

## Global Constraints

- Keep only the public innovation names `TCAA` and `ACED`.
- Do not add inference layers or inference-time computation.
- Do not hard-code `Surface_Crack`, `Delamination`, or class indices.
- Preserve all current TCAA centreline and crowd-topology behavior.
- Preserve the standard YOLO loss output shape and training API.
- Work directly in the local checkout because this directory is not recognized as a Git repository.

---

### Task 1: Residual-persistent TCAA

**Files:**
- Modify: `ultralytics/utils/tal.py`
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_tcaa_aced_dfl.py`

**Interfaces:**
- Consumes: current detached `pd_scores`, `pd_bboxes`, GT labels/boxes, candidate overlaps, topology density.
- Produces: `TopologyConflictAwareAssigner` with `_target_hardness`, `hardness_weight`, and `hardness_power`, while retaining the TCAA name and return contract.

- [ ] **Step 1: Write the failing residual-allocation test**

```python
def test_tcaa_allocates_more_late_stage_positives_to_the_unresolved_target():
    assigner = TopologyConflictAwareAssigner(
        topk=1, topk_extra=3, num_classes=1, hardness_weight=1.0, hardness_power=1.0
    )
    # Two square targets have equal topology. The first has accurate boxes/high confidence;
    # the second has poor overlap/low confidence. The unresolved target must receive more positives.
    mask_pos, _, _ = assigner.get_pos_mask(scores, boxes, labels, targets, anchors, valid)
    assert mask_pos[0, 1].sum() > mask_pos[0, 0].sum()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests\test_tcaa_aced_dfl.py::test_tcaa_allocates_more_late_stage_positives_to_the_unresolved_target -q
```

Expected: fail because `hardness_weight` and `hardness_power` are not accepted.

- [ ] **Step 3: Replace geometry-only dynamic top-k with residual-aware dynamic top-k**

Implement GT-class score gathering and detached per-target quality:

```python
class_scores = pd_scores.gather(
    2, gt_labels.squeeze(-1).long().unsqueeze(1).expand(-1, pd_scores.shape[1], -1)
).permute(0, 2, 1)
quality = torch.sqrt(overlaps.clamp(0, 1) * class_scores.clamp(0, 1))
best_quality = (quality * mask_in_gts).amax(-1)
hardness = (1.0 - best_quality).clamp(0, 1).pow(self.hardness_power) * mask_gt.squeeze(-1)
topology = (slender + crowd) * 0.5
extra_strength = (topology + self.hardness_weight * hardness).clamp(0, 1)
dynamic_topk = self.topk + torch.round(self.topk_extra * extra_strength).long()
```

Use the same hardness when resolving anchors shared by multiple GTs, so a still-unresolved target is not always displaced by an already-easy neighbour.

- [ ] **Step 4: Add configuration wiring**

Add:

```yaml
tcaa_hardness_weight: 1.0
tcaa_hardness_power: 1.0
```

Pass both values from `v8DetectionLoss` to `TopologyConflictAwareAssigner`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests\test_tcaa_aced_dfl.py -q
```

Expected: all TCAA tests pass.

### Task 2: Hardness-persistent ACED

**Files:**
- Modify: `ultralytics/nn/modules/battery.py`
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_tcaa_aced_dfl.py`

**Interfaces:**
- Consumes: raw DFL logits, anchor points, assigned grid-space boxes, positive mask.
- Produces: the existing `AspectConditionedEdgeDFL` loss with IoU-residual weighting and first/second-moment edge-pair calibration.

- [ ] **Step 1: Write the failing variance-calibration test**

```python
def test_aced_penalizes_diffuse_edges_even_when_the_expected_box_is_correct():
    sharp = logits_with_one_bin_per_edge()
    diffuse = logits_with_symmetric_bins_and_the_same_expected_edges()
    assert aced(diffuse, anchors, targets, foreground) > aced(sharp, anchors, targets, foreground)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests\test_tcaa_aced_dfl.py::test_aced_penalizes_diffuse_edges_even_when_the_expected_box_is_correct -q
```

Expected: fail because current ACED compares only expected edge positions.

- [ ] **Step 3: Replace target-score suppression with localization residual weighting**

Decode expected positive boxes, calculate detached IoU, and use:

```python
hardness = (1.0 - iou).detach().clamp(0, 1)
weights = self.hardness_floor + hardness.pow(self.hardness_power)
```

Remove `target_scores` from the ACED interface and call site.

- [ ] **Step 4: Add DFL second-moment calibration**

Calculate predicted edge variance and the ideal two-bin DFL target variance:

```python
pred_variance = (probabilities * (bins - expected.unsqueeze(-1)).square()).sum(-1)
fraction = target_edges - target_edges.floor()
target_variance = fraction * (1.0 - fraction)
variance_loss = smooth_l1(log1p(pred_pair_variance), log1p(target_pair_variance))
```

Apply the existing aspect-conditioned short-axis pair weights to both mean and variance terms.

- [ ] **Step 5: Add configuration wiring**

Add:

```yaml
aced_hardness_floor: 0.1
aced_hardness_power: 2.0
aced_variance_gain: 0.25
```

- [ ] **Step 6: Run focused and integration tests**

Run:

```powershell
D:\anaconda\envs\test\python.exe -m pytest tests\test_tcaa_aced_dfl.py -q
D:\anaconda\envs\test\python.exe -m compileall -q ultralytics train.py
```

Expected: all tests pass and compilation exits with code 0.

