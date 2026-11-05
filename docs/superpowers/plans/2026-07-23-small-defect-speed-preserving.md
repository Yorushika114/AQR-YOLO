# 电池表面小缺陷精度与速度平衡实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不增加完整检测分支的前提下，增强三个现有电池缺陷创新模块对针孔和细裂纹的特征利用能力，并控制训练速度开销。

**Architecture:** 在 `battery.py` 的 MDRH 中使用无参数局部高频能量调节已有形态路由；在 BCRH 辅助损失中加入归一化的逐尺度权重；在 CDAA 的自适应 top-k 中加入受上限约束的小目标面积因子。基线开关关闭时三项逻辑均不参与计算。

**Tech Stack:** Python 3.12、PyTorch、Ultralytics 8.4.94、pytest。

## Global Constraints

- 不新增完整 P2 检测头、Transformer、大型注意力或额外完整前向分支。
- 保持 `mdrh`、`bcrh`、`cdaa` 独立开关和干净 YOLO11 基线路径。
- 保持数据划分、随机种子、输入尺寸、batch、epoch、优化器和验证方式不变。
- 默认新增训练开销目标控制在当前训练时间的约 10% 以内，实际以远程复训测量为准。
- 当前数据集空标签不做删除、补标或转换。

---

### Task 1: 为三项精度增强添加失败测试

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`

- [ ] **Step 1: Write the failing tests**

新增以下行为测试：

```python
def test_mdrh_detail_energy_changes_local_routes():
    module = MorphologyDrivenRoutedHead(channels=(16, 32, 64), num_classes=3)
    smooth = [torch.ones(1, c, s, s) for c, s in ((16, 12), (32, 6), (64, 3))]
    detail = [x.clone() for x in smooth]
    detail[0][:, :, 5, 5] = 8.0
    module(smooth)
    smooth_route = module.last_routes[0].clone()
    module(detail)
    detail_route = module.last_routes[0]
    assert not torch.allclose(smooth_route, detail_route)


def test_bcrh_level_weights_are_normalized():
    outputs = [torch.zeros(1, 2, 12, 12), torch.zeros(1, 2, 6, 6), torch.zeros(1, 2, 3, 3)]
    batch = {"img": torch.zeros(1, 3, 24, 24), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "batch_idx": torch.tensor([0])}
    loss = box_contour_response_loss(outputs, batch, level_weights=[1.25, 1.0, 0.75])
    assert torch.isfinite(loss)


def test_cdaa_small_targets_receive_bounded_extra_topk():
    assigner = TaskAlignedAssigner(topk=10, num_classes=3, cdaa=True, class_counts=[1000, 100, 10], cdaa_small_object_power=0.20)
    labels = torch.tensor([[[0], [2]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.02, 0.02]]])
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    topk = assigner.compute_adaptive_topk(labels, boxes, mask)
    assert topk.shape == (1, 2)
    assert topk[0, 1] >= topk[0, 0]
    assert topk.max() <= 12 and topk.min() >= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: FAIL because the new detail route, BCRH level-weight argument, and CDAA small-object parameter do not yet exist.

### Task 2: 实现 MDRH 局部高频细节路由

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\modules\battery.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\cfg\default.yaml`

- [ ] **Step 1: Add the minimal implementation**

在 `MorphologyDrivenRoutedHead.__init__` 增加 `detail_weight` 参数，默认值为 `0.15`；在 `forward()` 中使用以下无参数细节计算，并将其加入已有路由：

```python
local_detail = x - torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
detail_energy = local_detail.abs().mean(dim=1, keepdim=True)
detail_energy = detail_energy / detail_energy.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)
detail_bias = torch.cat((detail_energy, detail_energy, detail_energy * 0.5), dim=1)
route = (global_route + local_route + self.detail_weight * detail_bias).softmax(dim=1)
```

保持 `branches` 仍为 point/line/region 三个分支，保持 `last_routes` 的形状为 `[B, 3, H, W]`。

- [ ] **Step 2: Run MDRH tests**

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: 新增细节路由测试和原有 MDRH 测试通过。

### Task 3: 实现 BCRH 多尺度轮廓监督

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\modules\battery.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\tasks.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\cfg\default.yaml`

- [ ] **Step 1: Add level weights without changing loss scale**

修改 `box_contour_response_loss(outputs, batch, dice_weight=0.25, level_weights=None)`：

```python
weights = outputs[0].new_tensor(level_weights or [1.0] * len(outputs))
weights = weights[: len(outputs)].clamp_min(0.0)
weights = weights / weights.sum().clamp_min(1e-6) * len(outputs)
for level_weight, output, target in zip(weights, outputs, targets):
    loss = loss + level_weight * ((1.0 - dice_weight) * bce + dice_weight * dice.mean())
```

当未传入配置时使用全 1 权重，保持旧行为。

- [ ] **Step 2: Plumb configuration**

在 `default.yaml` 增加 `bcrh_level_weights: [1.25, 1.0, 0.75]`，在 `DetectionModel._add_bcrh_loss()` 中读取并传入。

- [ ] **Step 3: Run BCRH tests**

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: BCRH 输出、Soft Dice、逐尺度权重和反向传播测试通过。

### Task 4: 实现 CDAA 小目标面积自适应分配

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\utils\tal.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\utils\loss.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\cfg\default.yaml`

- [ ] **Step 1: Add bounded small-object factor**

在 `TaskAlignedAssigner` 中增加 `cdaa_small_object_power=0.20`，并在 `compute_adaptive_topk()` 中将有效框面积与有效框面积中位数比较：

```python
areas = ((gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(1e-4) * (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(1e-4))
valid_areas = areas[valid]
median_area = valid_areas.median().clamp_min(1e-6) if valid_areas.numel() else areas.new_tensor(1.0)
small_factor = (median_area / areas.clamp_min(1e-6)).pow(self.cdaa_small_object_power).clamp(1.0, 1.75)
topk = (topk.float() * small_factor).round().clamp(self.cdaa_min_topk, self.cdaa_max_topk).long()
```

在 `cdaa=False` 时不调用该分支。

- [ ] **Step 2: Plumb configuration and run tests**

在 `loss.py` 传入 `cdaa_small_object_power`，在 `default.yaml` 增加同名配置，然后运行：

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: 小目标 top-k 不小于普通目标、仍受上下限约束，原有 CDAA 测试保持通过。

### Task 5: 联合连通性、速度和静态验证

**Files:**
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_validate_trained_model.py`

- [ ] **Step 1: Run regression tests**

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py tests/test_validate_trained_model.py -q -p no:cacheprovider
```

- [ ] **Step 2: Run AST/YAML validation**

验证所有修改的 Python 文件可以由 `ast.parse` 解析，`default.yaml` 可以由 YAML 解析器读取。

- [ ] **Step 3: Run fixed-size benchmark**

固定三层特征和随机 batch，分别测量 MDRH+BCRH 前向、BCRH 损失、CDAA 分配；不得出现数量级增加。远程服务器上的每轮耗时仍需用完整训练复测。

- [ ] **Step 4: Report experiment protocol**

远程复训保持 50 epoch、相同数据 YAML、相同 seed、batch、优化器、`imgsz=160` 和验证脚本，先比较基线、原三模块和增强后三模块；不得同时改变输入尺寸，否则无法判断新增模块收益。

