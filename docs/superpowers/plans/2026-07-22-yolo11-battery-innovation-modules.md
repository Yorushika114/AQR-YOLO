# YOLO11 电池表面缺陷三模块实现计划
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
**Goal:** 在 Ultralytics 8.4.94 的 YOLO11 检测链路中实现并连通 MDRH、BCRH、CDAA 三个默认关闭、可独立消融的研究模块，并完成轻量级代码连通性测试。
**Architecture:** MDRH 插入 Detect head 前的多尺度特征流，使用点/线/区域分支和路由权重增强形态表达；BCRH 复用 Detect 的多尺度特征，在训练期生成中心/边界响应并通过框标注构造辅助损失；CDAA 扩展 TaskAlignedAssigner，根据类别稀疏性和同图目标局部密度自适应分配 top-k。关闭开关时保持原始 YOLO11 前向、损失和赋值行为。
**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.94, pytest。

## 全局约束

- [ ] 所有新增配置默认为关闭：`mdrh=False`、`bcrh=False`、`cdaa=False`。
- [ ] 不改变现有 DIDC、MCRD、CBSC、MDSC 的语义和默认行为。
- [ ] 只使用目标检测框标注；BCRH 不要求掩码或分割标注。
- [ ] 先写失败测试并确认失败，再实现最小代码并让测试通过。
- [ ] 只做导入、模块前向、损失、赋值和基线关闭开关测试，不启动完整数据集训练。

## 任务 1：新增模块与基础单元测试

**文件：** `ultralytics/nn/modules/battery.py`、`ultralytics/nn/modules/__init__.py`、`tests/test_yolo11_innovation_modules.py`

- [ ] 编写失败测试：MDRH 接收三层特征并保持尺寸/通道；BCRH 输出每层中心/边界响应；关闭模块时输出与输入一致。
- [ ] 编写失败测试：BCRH 框目标响应损失可计算、有限且可反向传播。
- [ ] 实现 `MorphologyDrivenRoutedHead`：点、线、区域三分支，软路由融合，支持多尺度输入。
- [ ] 实现 `BoxContourResponseHead`：每层输出 2 通道响应图，并提供仅依赖框标注的中心/边界目标构造及损失函数。
- [ ] 导出新类/函数，确保 `from ultralytics.nn.modules import ...` 可用。
- [ ] 运行新增测试，确认基础模块通过。

## 任务 2：接入 Detect head 与训练损失

**文件：** `ultralytics/nn/modules/head.py`、`ultralytics/nn/tasks.py`、`ultralytics/models/yolo/detect/train.py`

- [ ] 编写失败测试：开启 MDRH/BCRH 后，YOLO11 Detect 训练前向仍返回 boxes/scores/feats，并带有 BCRH 辅助输出。
- [ ] 编写失败测试：DetectionModel 的 `configure_battery_modules()` 可按开关创建模块，重复调用不重复注册。
- [ ] 在 Detect 中增加可选模块配置和训练期 BCRH 输出；默认关闭时不增加前向分支。
- [ ] 在 DetectionModel 中提供配置入口，并在检测训练器设置模型属性后、构建优化器前完成配置。
- [ ] 在 `DetectionModel.loss` 中加入 BCRH 加权辅助损失，同时保持原有三项 loss_items 结构兼容日志系统。
- [ ] 运行模型构建、训练模式前向、损失反向和重复配置测试。

## 任务 3：接入 CDAA 赋值器

**文件：** `ultralytics/utils/tal.py`、`ultralytics/utils/loss.py`

- [ ] 编写失败测试：默认 CDAA 关闭时与原始 TaskAlignedAssigner 结果一致。
- [ ] 编写失败测试：开启 CDAA 后 rare/sparse 目标获得不低于普通/密集目标的有效 top-k，输出形状与原赋值器一致。
- [ ] 扩展 TaskAlignedAssigner，支持按类别频次和同图目标中心邻域密度计算每个 GT 的自适应 top-k。
- [ ] 在 `v8DetectionLoss` 中按模型参数选择 CDAA，并把可选类别统计/上下限传入。
- [ ] 运行赋值器形状、单元行为和 DetectionLoss 初始化测试。

## 任务 4：整链路验证与交付

**文件：** `tests/test_yolo11_innovation_modules.py`、必要时 `AGENTS.md` 或配置说明

- [ ] 运行新增测试和已有电池模块回归测试。
- [ ] 用小尺寸随机输入完成一次 YOLO11n 的 train/eval 前向，确认输出无 NaN、梯度可回传、开关独立。
- [ ] 检查默认关闭模型的输出结构与新模块开启前兼容。
- [ ] 汇总实际修改文件、测试命令、测试结果和当前未覆盖的完整训练验证项。
