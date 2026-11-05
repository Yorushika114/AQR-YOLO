# MCSR、MS-SW 与 T-MCRA 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留 MCSR，新增训练期 MS-SW 与检测头 T-MCRA，并移除已弃用的自定义缺陷模块及其配置入口。

**Architecture:** MCSR 继续在三尺度检测特征上执行零增益初始化的跨尺度残差路由。T-MCRA 从同一组形态描述符产生分类、回归两套独立的通道调制，使两条检测分支不再共享同一种特征偏置。MS-SW 不改写 TAL 的正样本集合，而对其软目标分数施加尺度一致性与形态难度权重，影响分类与框回归的监督强度。

**Tech Stack:** Python 3.12、PyTorch、Ultralytics 8.4.94、pytest。

## Global Constraints

- 直接修改当前本地检出目录；该目录不是 Git 仓库。
- 只保留 MCSR、MS-SW、T-MCRA 三个实验模块及它们的必要接线。
- 新模块必须默认关闭或以恒等映射初始化，保证加载 `yolo11n.pt` 时不扰动预训练头。
- 使用固定 `seed=0` 的 5 epoch 训练作筛选，最终结论不能替代 50 epoch 实验。

---

### Task 1: 增加 T-MCRA 并保留纯 MCSR

**Files:**
- Modify: `ultralytics/nn/modules/battery_next.py`
- Modify: `ultralytics/nn/modules/head.py`
- Test: `tests/test_ms_sw_tmcra.py`

**Interfaces:**
- Produces: `TaskModulatedMCSRAdapter(channels, descriptor_dim, hidden_dim)`，返回回归和分类两组尺度特征。
- Consumes: `MorphologyConditionedScaleRouter.describe(features)` 的批级形态描述符。

- [ ] **Step 1: 写出失败测试**

```python
def test_task_adapter_is_identity_at_initialization():
    adapter = TaskModulatedMCSRAdapter((8, 16), descriptor_dim=8, hidden_dim=4)
    features = [torch.randn(2, 8, 20, 20), torch.randn(2, 16, 10, 10)]
    reg, cls = adapter(features, torch.randn(2, 8))
    assert torch.equal(reg[0], features[0])
    assert torch.equal(cls[1], features[1])
```

- [ ] **Step 2: 运行失败测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q`

Expected: 因 `TaskModulatedMCSRAdapter` 未定义而失败。

- [ ] **Step 3: 实现最小模块与头接线**

```python
task_logits = self.task_router(descriptor).split(self.channels, dim=1)
scale = 1.0 + self.gain * torch.tanh(task_logit).unsqueeze(-1).unsqueeze(-1)
return [feature * scale for feature, scale in zip(features, scales)]
```

`AdaptiveDefectDetect` 以 MCSR 后的特征分别调用原生 `cv2`、`cv3`，并维持原生 `Detect` 的输出字典和推理路径。

- [ ] **Step 4: 运行模块测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q`

Expected: 新增 T-MCRA 恒等初始化和梯度测试通过。

### Task 2: 接入 MS-SW 软监督

**Files:**
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_ms_sw_tmcra.py`

**Interfaces:**
- Produces: `MorphologyScaleSoftWeighting(target_bboxes, target_scores, fg_mask, stride_tensor)`。
- Consumes: TAL 已分配的软目标；不改变 `fg_mask` 或 `target_gt_idx`。

- [ ] **Step 1: 写出失败测试**

```python
def test_ms_sw_preserves_background_and_reweights_foreground_only():
    weighted = MorphologyScaleSoftWeighting(...)(scores, boxes, foreground, strides)
    assert torch.equal(weighted[~foreground], scores[~foreground])
    assert not torch.equal(weighted[foreground], scores[foreground])
```

- [ ] **Step 2: 运行失败测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q`

Expected: 因 MS-SW 代码不存在而失败。

- [ ] **Step 3: 实现最小加权**

```python
span = (box_width * box_height).sqrt() / anchor_stride
scale_prior = torch.exp(-0.5 * ((span.log() - reference_span.log()) / temperature).square())
elongation = (long_side / short_side.clamp_min(eps)).log1p()
weight = (scale_prior / per_target_mean).clamp(min_weight, max_weight) * (1 + shape_gain * elongation)
return target_scores * weight.unsqueeze(-1)
```

该权重仅应用于 `fg_mask` 内的 TAL 软分数，随后按加权后的 `target_scores_sum` 计算 BCE、IoU 和 DFL。

- [ ] **Step 4: 运行模块测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q`

Expected: 背景不变、前景非硬性重加权、空前景安全返回的测试通过。

### Task 3: 清除弃用模块和旧训练入口

**Files:**
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/cfg/default.yaml`
- Modify: `ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml`
- Modify: `train.py`
- Modify: `adaptive_ablation.py`
- Delete: `ultralytics/utils/dcgp.py`
- Delete: `tests/test_dcgp.py`
- Delete: `ultralytics/cfg/models/11/yolo11n-p2-adaptive-defect.yaml`
- Delete: `ultralytics/cfg/models/11/yolo11n-p2-direct-defect.yaml`

**Interfaces:**
- Produces: 仅识别 `mcsr`、`tmcra`、`ms_sw` 及其超参数的训练配置。
- Removes: DTRM、OUDM、DCGP、局部竞争 MCSR、P2 头以及旧实验模块的运行入口。

- [ ] **Step 1: 写出失败清理测试**

```python
def test_custom_config_rejects_removed_experiment_arguments():
    with pytest.raises(SyntaxError):
        get_cfg(DEFAULT_CFG_DICT, {"dtrm": True})
```

- [ ] **Step 2: 运行失败测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q`

Expected: 当前 `dtrm` 仍是合法参数，因此断言失败。

- [ ] **Step 3: 删除旧模块接线与参数**

移除 `default.yaml` 中过时实验参数、`loss.py` 中的旧辅助损失/分配器分支、检测头和模型解析器中的旧自定义头入口。消融脚本只保留 `mcsr`、`mcsr_ms_sw`、`mcsr_tmcra`、`mcsr_ms_sw_tmcra` 四种固定种子配置。

- [ ] **Step 4: 运行清理测试和编译检查**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q; D:\anaconda\envs\test\python.exe -m compileall -q ultralytics train.py adaptive_ablation.py`

Expected: 测试全部通过，编译命令退出码为 0。

### Task 4: 连通性与消融筛选

**Files:**
- Test: `tests/test_ms_sw_tmcra.py`
- Run: `adaptive_ablation.py`

- [ ] **Step 1: 新配置的权重加载和前向测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_ms_sw_tmcra.py -q`

Expected: `yolo11n.pt` 能装载新 YAML，并产生三尺度的原生检测输出。

- [ ] **Step 2: 运行固定种子 5 epoch 消融**

Run: `D:\anaconda\envs\test\python.exe adaptive_ablation.py`

Expected: 输出 MCSR 基线、MS-SW 单开、T-MCRA 单开及共同开启的每类 mAP50-95 和运行目录。

- [ ] **Step 3: 对比并记录结果**

以同一数据划分、同一初始化、同一训练超参数对比三类别 mAP50-95；若一个模块单开同时降低 Surface_Crack 与 Delamination，则停止将其视为候选并回到根因分析。
