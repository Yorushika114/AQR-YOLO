# 电池表面缺陷三模块精度增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持约 1 分钟/轮训练速度的前提下，增强 MDRH、BCRH、CDAA 对 Surface_Crack 定位和分类的有效性。

**Architecture:** 保留三个创新模块的职责边界。MDRH 在已有点/线/区域分支上增加无卷积的逐位置路由；BCRH 将共享门控改为逐尺度门控并加入轻量软 Dice 监督；CDAA 在开启时增加预测框与真实框的长宽比一致性项。关闭模块时恢复原 YOLO11 路径。

**Tech Stack:** Python 3.12、PyTorch、Ultralytics 8.4.94、pytest。

## Global Constraints

- 不替换 MDRH、BCRH、CDAA 的研究定义，不新增 P2 检测层、大型注意力模块或额外前向分支。
- 保持数据集划分、输入尺寸、batch、优化器、epoch 和验证流程不变。
- `mdrh=False`、`bcrh=False`、`cdaa=False` 时必须保持基线行为。
- 训练速度应保持在当前约 1 分钟/轮附近。
- 所有精度变化必须通过相同训练预算和消融实验验证。

---

### Task 1: 增加精度增强行为的失败测试

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`
- Test: same file

**Interfaces:**
- Consumes: `MorphologyDrivenRoutedHead`, `BoxContourResponseHead`, `TaskAlignedAssigner`。
- Produces: 后续实现必须满足的局部路由、逐尺度门控、Dice 损失和形态感知分配接口。

- [ ] **Step 1: Write the failing tests**

新增以下断言：

```python
assert route.shape == (batch_size, 3, height, width)
assert torch.allclose(route.sum(dim=1), torch.ones_like(route[:, 0]), atol=1e-5)
assert module.gate_logit.shape == (3,)
assert torch.isfinite(box_contour_response_loss(outputs, batch)).all()
assert shape_quality.shape == alignment_metric.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: FAIL because current MDRH only exposes global route，BCRH 使用标量门控，CDAA 没有形态一致性项。

### Task 2: 实现 MDRH 局部形态路由

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\modules\battery.py`
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: 三个形态分支输出 `branches`，形状为 `[B, 3, C, H, W]`。
- Produces: `MorphologyDrivenRoutedHead.last_routes`，包含每个检测层一个 `[B, 3, H, W]` 路由图，并返回与输入相同形状的特征列表。

- [ ] **Step 1: Write the minimal implementation**

在 `forward()` 中保留现有全局路由，同时用已有分支的通道平均绝对响应构造局部路由：

```python
global_route = self.router[i](x)
local_route = branches.abs().mean(dim=2)
route = (global_route + local_route).softmax(dim=1)
self.last_routes.append(route.detach())
fused = (branches * route.unsqueeze(2)).sum(dim=1)
```

不增加新的卷积层，继续使用现有有界残差门控。

- [ ] **Step 2: Run MDRH tests**

Run:

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: MDRH 路由测试通过，输出形状不变。

### Task 3: 实现 BCRH 逐尺度门控和软 Dice 监督

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\modules\battery.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\modules\head.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\nn\tasks.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\cfg\default.yaml`
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: BCRH 的两个响应图和 `_box_response_targets()` 生成的软中心/边界目标。
- Produces: `BoxContourResponseHead.gate_logit` 形状 `[nl]`；`box_contour_response_loss(outputs, batch, dice_weight)` 返回标量损失。

- [ ] **Step 1: Write the minimal implementation**

将 `gate_logit` 从标量改成 `len(channels)` 个参数，在 `apply_feature_gate()` 中按特征层取对应门控；新增软 Dice：

```python
scale = self.gate_logit[i].tanh()
gated.append(feature + scale * feature * evidence)
```

Dice 项仅作用于中心和边界响应，不改变检测框损失。

- [ ] **Step 2: Add configuration plumbing**

在 `default.yaml` 增加 `bcrh_dice_weight: 0.25`，由 `DetectionModel._add_bcrh_loss()` 传入损失函数；未配置时默认 `0.25`。

- [ ] **Step 3: Run BCRH tests**

Run:

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: BCRH 的输出、门控初始化、Dice 损失和反向传播测试通过。

### Task 4: 实现 CDAA 形态感知任务分配

**Files:**
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\utils\tal.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\utils\loss.py`
- Modify: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\ultralytics\cfg\default.yaml`
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`

**Interfaces:**
- Consumes: 已筛选的 `pd_boxes`、`gt_boxes`、IoU 和类别得分。
- Produces: `TaskAlignedAssigner(cdaa_shape_weight=0.20)`；仅在 `cdaa=True` 时改变 `align_metric`。

- [ ] **Step 1: Write the minimal implementation**

对候选预测框和 GT 计算长宽比一致性：

```python
pred_ar = (pd_boxes[..., 2] / pd_boxes[..., 3].clamp_min(1e-4)).clamp_min(1e-4)
gt_ar = (gt_boxes[..., 2] / gt_boxes[..., 3].clamp_min(1e-4)).clamp_min(1e-4)
shape_quality = torch.exp(-(pred_ar.log() - gt_ar.log()).abs())
align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
if self.cdaa:
    align_metric = align_metric * shape_quality.pow(self.cdaa_shape_weight)
```

新增配置 `cdaa_shape_weight: 0.20`，关闭 CDAA 时不执行该计算。

- [ ] **Step 2: Run CDAA tests**

Run:

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py -q -p no:cacheprovider
```

Expected: 细长框候选的形态一致性分数更高，赋值张量形状不变。

### Task 5: 联合连通性和速度验证

**Files:**
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_yolo11_innovation_modules.py`
- Test: `D:\个人文件整理\02_开发项目\ultralytics-8.4.94\tests\test_validate_trained_model.py`

- [ ] **Step 1: Run module and validation tests**

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m pytest tests/test_yolo11_innovation_modules.py tests/test_validate_trained_model.py -q -p no:cacheprovider
```

Expected: 全部通过。

- [ ] **Step 2: Run a fixed-size forward/loss benchmark**

使用固定随机 batch，分别测试基线和三模块增强模型的 forward、loss、backward 时间；增强版本不得出现数量级增加。

- [ ] **Step 3: Run static syntax validation**

```powershell
& 'D:\anaconda\envs\test\python.exe' -B -m compileall -q ultralytics tests
```

Expected: 命令成功退出。

- [ ] **Step 4: Report remote experiment protocol**

远程服务器固定数据集、随机种子、batch、imgsz、epoch、optimizer 和验证脚本，只比较基线、原三模块和增强三模块三组结果，并记录每轮耗时。
