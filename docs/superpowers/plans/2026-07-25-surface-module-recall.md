# Surface Module Recall Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MDRH、BCRH、CDAA 内部加入只针对 `Surface_Crack` 的受限形态增强，以提高召回和 mAP50-95，同时保护 Pinhole 与 Delamination。

**Architecture:** 保持现有 YOLO11 检测头和损失接口不变。MDRH 只在最高分辨率层加入有界 Surface 细节路由；BCRH 只改变 Surface 的辅助响应目标；CDAA 只改变 Surface 的正样本候选上限/下限。关闭任一模块时，恢复现有路径。

**Tech Stack:** Python 3.12, PyTorch, Ultralytics 8.4.94, pytest, PowerShell.

## Global Constraints

- 数据集固定为 `D:\浏览器下载位置\detection_yolo_v2`，不修改旧的 `archive\detection_yolo_v2`。
- 不增加分割掩码，不修改标签，不改变类别编号。
- Surface 专用增益必须有上下限；Pinhole 和 Delamination 不使用 Surface 专用分支。
- 训练速度额外下降目标不超过约 20%。
- 当前项目 `.git` 目录为空，不能执行真实提交；保留所有改动在本隔离副本并记录回退文件。

---

### Task 1: 为 Surface 专用 MDRH 路由建立回归测试

**Files:**
- Modify: `tests/test_yolo11_innovation_modules.py`
- Test: `tests/test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: `MorphologyDrivenRoutedHead(channels, num_classes, focus_class, focus_line_gain, focus_cls_gain)`。
- Produces: 可验证的 Surface-only 高分辨率路由行为。

- [ ] **Step 1: 写失败测试**

增加以下测试，先要求模块在高分辨率层对局部细节的 Surface 路由增益大于平坦区域，并要求非 Surface 分类 logit 不变：

```python
def test_mdrh_surface_route_is_bounded_and_highres_only():
    from ultralytics.nn.modules import MorphologyDrivenRoutedHead

    module = MorphologyDrivenRoutedHead(
        channels=(16, 32, 64), num_classes=3, focus_class=1, focus_line_gain=0.12, focus_cls_gain=0.10
    )
    features = [torch.zeros(1, c, s, s) for c, s in ((16, 16), (32, 8), (64, 4))]
    features[0][:, :, 6:10, 3:13] = 4.0
    module(features)
    high = module.last_focus_maps[0][0, 0, 8, 8]
    flat = module.last_focus_maps[0][0, 0, 1, 1]
    assert high > flat
    assert float(module.residual_logit.sigmoid().max()) < 0.5

    scores = [torch.zeros(1, 3, s * s) for s in (16, 8, 4)]
    focused = module.apply_classification_focus(scores[0], level=0)
    assert torch.any(focused[:, 1] > scores[0][:, 1])
    assert torch.allclose(focused[:, 0], scores[0][:, 0])
    assert torch.allclose(focused[:, 2], scores[0][:, 2])
    assert torch.allclose(module.apply_classification_focus(scores[1], level=1), scores[1])
```

- [ ] **Step 2: 运行失败测试**

运行：

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py::test_mdrh_surface_route_is_bounded_and_highres_only -q
```

预期：新测试在当前实现上可能因默认路由增益不足而失败，记录失败原因后再改实现。

- [ ] **Step 3: 最小实现**

在 `ultralytics/nn/modules/battery.py` 中保留现有三分支结构，但将最高分辨率 Surface 路由改成有界的局部细节门控：

```python
surface_detail = focus_prior * (detail_energy / (1.0 + detail_energy))
surface_direction = focus_prior * (directional_energy / (1.0 + directional_energy))
focus_line_bias = _focus_line_route_bias(
    focus_prior,
    (0.65 * detail_energy + 0.35 * directional_energy).clamp(0.0, 4.0),
    self.focus_line_gain,
)
focus_map = (0.7 * surface_detail + 0.3 * surface_direction).clamp(0.0, 1.0)
```

将分类残差改为 `focus_cls_gain * focus_map`，保持只写入 `self.focus_class` 的切片；不改变 `x` 的形状和其他类别的 logit。

- [ ] **Step 4: 运行测试确认通过**

运行同一条 pytest 命令，预期输出 `1 passed`；再运行已有 MDRH 测试：

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py -k mdrh -q
```

- [ ] **Step 5: 记录可回退文件**

保存 `ultralytics/nn/modules/battery.py` 和测试文件的修改前副本到 `docs/superpowers/rollback/2026-07-25/`，不覆盖原始代码。

### Task 2: 让 BCRH 对 Surface 使用安全响应目标

**Files:**
- Modify: `ultralytics/nn/modules/battery.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `ultralytics/cfg/default.yaml`
- Modify: `tests/test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: `batch["cls"]`, `batch["bboxes"]`, 多尺度 BCRH 输出。
- Produces: `box_contour_response_loss(..., focus_class=1, surface_mode="center_band")`，且其他类别继续使用原边界目标。

- [ ] **Step 1: 写失败测试**

新增测试，验证同一个矩形框在 Surface 类别和 Delamination 类别下的 BCRH 第二通道目标不同，且两者都有限：

```python
def test_bcrh_surface_target_avoids_outer_box_boundary_only():
    from ultralytics.nn.modules.battery import _box_response_targets

    batch = {
        "img": torch.zeros(2, 3, 32, 32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.6, 0.6]]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    targets = _box_response_targets(batch, [(32, 32)], "cpu", torch.float32, focus_class=1)
    surface, region = targets[0][0, 1], targets[0][1, 1]
    assert torch.isfinite(targets[0]).all()
    assert surface[16, 16] > 0
    assert region[1, 1] > region[16, 16]
```

- [ ] **Step 2: 运行失败测试**

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py::test_bcrh_surface_target_avoids_outer_box_boundary_only -q
```

预期：当前 `_box_response_targets` 不接受 `focus_class`，测试失败。

- [ ] **Step 3: 实现类别感知目标**

修改 `_box_response_targets` 签名：

```python
def _box_response_targets(batch, sizes, device, dtype, focus_class=1):
```

在现有 `contour_target` 计算后增加 Surface 中心带目标：

```python
classes = batch.get("cls", boxes.new_full((number_of_boxes, 1), -1)).view(-1).long().to(device)
focus = classes.eq(int(focus_class))
normalized_distance = ((xx[None] - center_x[:, None, None]) / half_w[:, None, None].clamp_min(1)).abs() + (
    (yy[None] - center_y[:, None, None]) / half_h[:, None, None].clamp_min(1)
).abs()
surface_band = inside & (normalized_distance <= 0.9)
contour_target = torch.where(focus[:, None, None], surface_band, contour_target).to(dtype)
```

把 `focus_class` 从 `_add_bcrh_loss` 传入 `box_contour_response_loss`，并从配置读取 `bcrh_focus_class`。默认仍为 1。

- [ ] **Step 4: 更新配置并测试**

在 `ultralytics/cfg/default.yaml` 增加：

```yaml
bcrh_focus_class: 1
```

运行：

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py -k bcrh -q
```

预期：所有 BCRH 测试通过，输出仍保持标量损失和三项 YOLO 损失向量接口。

### Task 3: 增加 CDAA 的 Surface 受限正样本下限

**Files:**
- Modify: `ultralytics/utils/tal.py`
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Modify: `tests/test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: `gt_labels`, `gt_bboxes` 和当前 CDAA 的对齐指标。
- Produces: 只对 `cdaa_focus_class` 生效的、具有上下限的 adaptive top-k 和 target score floor。

- [ ] **Step 1: 写失败测试**

新增：

```python
def test_cdaa_surface_topk_has_bounded_recall_floor():
    from ultralytics.utils.tal import TaskAlignedAssigner

    assigner = TaskAlignedAssigner(
        topk=10, num_classes=3, stride=[8, 16, 32], cdaa=True,
        cdaa_max_topk=12, cdaa_focus_class=1, cdaa_focus_min_topk=6, cdaa_focus_max_topk=12,
    )
    labels = torch.tensor([[[1], [0], [2]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.50, 0.05], [0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]])
    mask = torch.ones(1, 3, 1, dtype=torch.bool)
    topk = assigner.compute_adaptive_topk(labels, boxes, mask)
    assert 6 <= int(topk[0, 0]) <= 12
    assert int(topk[0, 1]) <= 10
    assert int(topk[0, 2]) <= 10
```

- [ ] **Step 2: 运行失败测试**

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py::test_cdaa_surface_topk_has_bounded_recall_floor -q
```

预期：当前 `TaskAlignedAssigner` 没有 `cdaa_focus_min_topk` 参数，测试失败。

- [ ] **Step 3: 实现受限 Surface top-k**

在 `TaskAlignedAssigner.__init__` 增加两个参数并保存：

```python
cdaa_focus_min_topk: int | None = None,
cdaa_focus_max_topk: int | None = None,
```

在 `compute_adaptive_topk` 生成基础 top-k 后，只对 `gt_labels == cdaa_focus_class` 应用：

```python
focus = gt_labels.squeeze(-1).eq(self.cdaa_focus_class)
focus_topk = adaptive_topk.clamp(
    min=int(self.cdaa_focus_min_topk),
    max=int(self.cdaa_focus_max_topk),
)
adaptive_topk = torch.where(focus, focus_topk, adaptive_topk)
```

并确保普通类别仍被 `self.topk` 上限约束。把配置字段从 `default.yaml` 传入现有 assigner 构造路径。

在 `ultralytics/utils/loss.py` 的 `TaskAlignedAssigner(...)` 构造调用中显式传入：

```python
cdaa_focus_min_topk=getattr(h, "cdaa_focus_min_topk", None),
cdaa_focus_max_topk=getattr(h, "cdaa_focus_max_topk", None),
```

- [ ] **Step 4: 运行 CDAA 测试**

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py -k cdaa -q
```

预期：所有 CDAA 测试通过，且普通类别候选数不受 Surface 专用参数影响。

### Task 4: 集成测试、最小训练和性能核验

**Files:**
- Modify: `battery_defect_055_config.yaml`
- Test: `tests/test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: 三个已完成模块和新数据集配置。
- Produces: 可运行的模型构建、单 batch 训练、单次验证和时间/显存记录。

- [ ] **Step 1: 更新稳定配置**

在 `battery_defect_055_config.yaml` 增加：

```yaml
bcrh_focus_class: 1
cdaa_focus_min_topk: 6
cdaa_focus_max_topk: 12
```

保留 `mdrh: true`、`bcrh: true`、`cdaa: true`，不打开 `didc`、`mcrd`、`mdsc` 或 `cbsc`。

- [ ] **Step 2: 运行完整模块测试**

```powershell
& 'D:\anaconda\envs\test\python.exe' -m pytest tests/test_yolo11_innovation_modules.py tests/test_battery_modules.py -q
```

预期：全部通过；若出现旧测试因签名变更失败，优先补充默认参数而不是修改测试语义。

- [ ] **Step 3: 做 CPU 最小连通性测试**

```powershell
$env:PYTHONPATH='D:\个人文件整理\02_开发项目\ultralytics-8.4.94-surface-sandbox'
& 'D:\anaconda\envs\test\python.exe' -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); r=m.train(data='D:/浏览器下载位置/detection_yolo_v2/data.yaml', epochs=1, imgsz=160, batch=2, workers=0, device='cpu', mdrh=True, bcrh=True, cdaa=True, name='surface_smoke', project='runs/surface_smoke', save=False, plots=False); print('SMOKE_OK')"
```

预期：完成 1 个 epoch 并打印 `SMOKE_OK`；该测试只验证连通性，不作为精度结论。

- [ ] **Step 4: 运行新数据集验证**

使用相同权重、相同 `imgsz`、相同 batch 和随机种子，对比旧稳定配置与新配置，记录每个类别 `P/R/mAP50/mAP50-95`、每轮训练时间、显存和验证耗时。只有 Surface 提升且另外两类未明显下降时，才建议将改动回写原项目。

## 回退记录

如果失败，按以下顺序回退：先关闭 `bcrh` 的 Surface 中心带目标，再恢复 CDAA 的默认 top-k，最后恢复 `battery.py` 和 `tal.py` 的修改。不得删除新数据集或旧实验输出。
