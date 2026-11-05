# P2 与 Surface 动态回归实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为细长、低像素占比的 `Surface_Crack` 增加 P2/4 高分辨率检测分支，并用独立的动态 IoU 回归权重提升低质量裂缝框的召回，保持 `Pinhole` 与 `Delamination` 路径不变。

**Architecture:** 在 YOLO11 原有 P3/8、P4/16、P5/32 检测头前增加 P2/4 特征融合，检测输出变为 P2-P5 四尺度。回归损失只在 `target_scores.argmax(-1)==surface_focus_class` 的正样本上应用动态 IoU 权重，其余类别沿用原始 CIoU 路径。P2 模型和实验参数通过新 YAML/配置文件启用，不覆盖稳定配置。

**Tech Stack:** Python 3.11/3.12、PyTorch、Ultralytics 8.4.94、PyYAML、pytest。

## Global Constraints

- 不删除或覆盖现有稳定配置和权重。
- 所有新逻辑必须可独立关闭，默认关闭动态回归；P2 由模型 YAML 选择。
- Surface 类索引固定为 1；Pinhole 和 Delamination 不增加额外回归权重。
- 训练配置使用 `imgsz=960`、`epochs=80`，实际 batch 由显存调整。
- 必须通过 YAML 构建、随机前向/反传、最小训练验证。

### Task 1: 为 P2 检测头建立回归测试

**Files:**
- Create: `ultralytics/cfg/models/11/yolo11-p2.yaml`
- Test: `tests/test_yolo11_p2_surface.py`

- [ ] 测试模型 YAML 能构建，并且 Detect 接收四个尺度。
- [ ] 测试四尺度模型在 64×64 随机输入上可前向。
- [ ] 先运行测试，确认新 YAML 不存在时按预期失败。

### Task 2: 实现 P2/4 模型结构

**Files:**
- Create: `ultralytics/cfg/models/11/yolo11-p2.yaml`

- [ ] 复用 YOLO11 backbone 的 P2/4 特征，插入 P2 上采样融合，并依次重建 P3/P4/P5。
- [ ] Detect 输入顺序固定为 `[P2, P3, P4, P5]`。
- [ ] 运行 Task 1 测试和 `py_compile`。

### Task 3: 为 Surface 动态 IoU 回归建立回归测试

**Files:**
- Modify: `ultralytics/utils/loss.py`
- Test: `tests/test_yolo11_p2_surface.py`

- [ ] 测试 `BboxLoss(surface_wiou=True, focus_class=1)` 对 Surface 正样本产生有限、可反传的损失。
- [ ] 测试同一批次中非 Surface 类回归权重不变。
- [ ] 测试开关关闭时与原始回归路径等价。
- [ ] 先运行测试，确认新参数不存在时按预期失败。

### Task 4: 接入动态回归开关

**Files:**
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `train.py`
- Modify: `ultralytics/cfg/default.yaml`

- [ ] 在 `BboxLoss` 中增加 `surface_wiou=False`、`surface_wiou_focus_class=1` 参数。
- [ ] 仅根据正样本目标类别对 Surface 应用有界动态权重，Pinhole/Delamination 使用原始权重。
- [ ] 将配置参数从训练入口传到 `v8DetectionLoss`。
- [ ] 保持旧配置的默认行为不变。

### Task 5: 新增长训实验配置并同步

**Files:**
- Create: `battery_defect_p2_surface_config.yaml`
- Modify: `tests/test_yolo11_p2_surface.py`

- [ ] 配置 `model` 指向 `yolo11-p2.yaml`，`imgsz=960`、`epochs=80`、模块全开、`surface_wiou=true`。
- [ ] 配置独立 `project/name`，不覆盖旧实验目录。
- [ ] 在测试中校验关键开关、尺寸、训练轮数和模型路径。

### Task 6: 完整验证

- [ ] 运行 `pytest tests/test_yolo11_p2_surface.py tests/test_yolo11_innovation_modules.py tests/test_battery_modules.py -q`。
- [ ] 运行全部改动文件的 `py_compile`。
- [ ] 使用新 YAML 做 1 epoch、极小数据比例的 CPU smoke train/val。
- [ ] 将已验证文件显式同步回原项目，并在原项目重复测试。

