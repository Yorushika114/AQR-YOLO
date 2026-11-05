# TCAA Anchor Competition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Make TCAA alter dense-target candidate ranking through an anchor-specific competition score, while leaving ACED unchanged.

**Architecture:** For each valid ground-truth target and candidate anchor, compare the anchor's distance to the current target centre with its distance to the nearest neighbouring target centre.  In dense regions, only anchors geometrically more affiliated with the current target receive a ranking bonus.  This replaces the existing per-target crowd scalar, which cannot reorder candidates within that target.

**Tech Stack:** Python, PyTorch, Ultralytics detection loss tests.

## Global Constraints

- Keep TCAA and ACED train-only and category agnostic.
- Preserve all existing configuration flags and ACED behavior.
- Make no dataset, model-YAML, or inference/NMS changes.
- Validate the new candidate ordering with a deterministic unit test before integration.

---

### Task 1: Prove anchor-specific competition changes ranking

**Files:**
- Modify: `tests/test_tcaa_aced_dfl.py`
- Modify: `ultralytics/utils/tal.py`

**Interfaces:**
- Produces: `TopologyConflictAwareAssigner._crowd_affinity(anc_points, gt_bboxes, mask_gt)` returning `(B, G, A)` values in `[0, 1]`.
- Consumes: anchor centres `(A, 2)`, target boxes `(B, G, 4)`, valid-target mask `(B, G, 1)`.

- [ ] Write a test with two close targets and equal task-aligned scores.  Assert that the anchor near target 0 has a larger affinity for target 0, while the anchor near target 1 has a larger affinity for target 1.
- [ ] Run only the new test and confirm it fails because `_crowd_affinity` does not exist.
- [ ] Implement `_crowd_affinity` using normalized squared centre distances and the nearest valid competing target.  Set isolated-target affinity to zero so their ranking remains baseline-equivalent.
- [ ] Update TCAA topology ranking so the crowd contribution is `crowd_density * crowd_affinity`, not `crowd_density` alone.
- [ ] Run the focused test and then all TCAA/ACED tests.

### Task 2: Verify train-loss wiring without changing ACED

**Files:**
- Modify: `tests/test_tcaa_aced_dfl.py`
- Modify: `ultralytics/utils/tal.py`

**Interfaces:**
- Consumes: existing `TopologyConflictAwareAssigner.get_pos_mask`.
- Produces: the same assignment return values and unchanged configuration API.

- [ ] Add a test that checks dense neighbouring targets receive a different topology-metric ordering from their raw alignment ordering.
- [ ] Run it and confirm it fails against the old scalar-only ranking behavior.
- [ ] Keep the existing dynamic-positive and quality-gating logic unchanged; only apply the new anchor-specific competition factor inside the crowd ranking term.
- [ ] Run the full focused test module, Python compilation, and a CPU smoke backward pass.

### Task 3: Controlled external validation

**Files:**
- No code changes.

- [ ] Train exactly 5 epochs with the existing `train.py` configuration and fixed seed.
- [ ] Compare against the latest `train-2` run using all, Surface_Crack, and Delamination `mAP50-95` plus `R`.
- [ ] Only tune an additional parameter after this single-variable comparison; do not modify ACED in the same experiment.
