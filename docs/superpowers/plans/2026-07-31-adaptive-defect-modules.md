# 自适应电池缺陷检测模块实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**目标：** 在彻底移除旧 MTCRL、GUPMM、BESR、SGH 等结构后，基于原生 YOLO11n 实现三个可独立消融的新模块，并优先提升 Surface_Crack 与 Delamination 的最终 mAP50-95。

**架构：** 使用原生 YOLO11n 作为严格基线。DTRM 和 MCSR 作为检测头前的可选特征处理模块；OUDM 保留原生 Task-Aligned Assigner，仅在检测损失中对 DFL 分布的不确定性进行低权重校准，避免改变正样本分配逻辑。

**技术栈：** PyTorch、Ultralytics YOLO11、pytest、现有 YOLO YAML 解析器。

## 全局约束

- 旧模块 MTCRL、GUPMM、BESR、SGH、PCRA、SRF、APCC、TCAA、ACED 默认全部关闭。
- 原生 YOLO11n 配置必须能单独构建并作为严格基线。
- 新模块必须分别支持开关，并保持关闭时与原生检测输出形状一致。
- DTRM 与 MCSR 使用零初始化残差，避免模型刚开始训练时破坏预训练特征。
- OUDM 不替换原生 Task-Aligned Assigner，初始损失权重保持较低。
- 每个新模块都必须有单元测试、模型构建测试和最小前向/反向测试。

## 任务 1：建立新模块失败测试

**文件：**

- 新建：`tests/test_adaptive_defect_modules.py`
- 参考：`tests/test_final_ceiling_modules.py`

- [ ] 测试 DTRM 的关闭路径、输出形状和零初始化残差。
- [ ] 测试 MCSR 的关闭路径、输出形状和零初始化残差。
- [ ] 测试 OUDM 对有效前景和空前景都返回有限标量，并能反向传播。
- [ ] 测试 `yolo11n-adaptive-defect.yaml` 可以构建，且检测头输出维度与原生 YOLO11n 一致。
- [ ] 先运行测试，确认因新模块尚不存在而失败。

## 任务 2：实现 DTRM 与 MCSR

**文件：**

- 新建：`ultralytics/nn/modules/battery_next.py`
- 修改：`ultralytics/nn/modules/__init__.py`

- [ ] 实现 `DefectTopologyRelationModule`：在每个检测尺度上使用局部 3×3 稀疏关系，结合特征相似度和邻接方向形成消息传递，输出使用零初始化投影残差。
- [ ] 实现 `MorphologyConditionedScaleRouter`：从每个尺度提取均值、方差和横纵变化统计，生成目标尺度到源尺度的归一化动态路由权重，并以零初始化残差融合。
- [ ] 导出两个类，避免修改现有旧模块实现。
- [ ] 运行新模块单元测试并确认通过。

## 任务 3：实现自适应检测头与 OUDM

**文件：**

- 修改：`ultralytics/nn/modules/head.py`
- 修改：`ultralytics/nn/tasks.py`
- 修改：`ultralytics/utils/loss.py`
- 修改：`ultralytics/cfg/default.yaml`

- [ ] 新增 `AdaptiveDefectDetect`，继承原生 `Detect`，只负责可选调用 DTRM/MCSR 后执行原生检测分支。
- [ ] 新增 `set_adaptive_modules_enabled()`，由损失初始化阶段读取训练开关。
- [ ] 新增 `OverlapUncertaintyDistributionMatcher`，只校准 DFL 分布熵与预测框重叠质量，不修改原生正样本分配器。
- [ ] 在 `v8DetectionLoss` 中增加 `oudm` 和 `oudm_loss_gain` 配置，并保持关闭时原损失完全不变。
- [ ] 将新检测头加入安全模型解析器。

## 任务 4：创建严格基线和实验配置

**文件：**

- 新建：`ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml`
- 修改：`train.py`

- [ ] 新 YAML 完全复制原生 YOLO11 结构，仅将最后检测头替换为 `AdaptiveDefectDetect`。
- [ ] `train.py` 使用新 YAML 加载 `yolo11n.pt`，并将所有旧模块开关显式设为 `False`。
- [ ] 新模块开关默认设为独立可控，首轮默认只打开 DTRM、MCSR，OUDM 使用低权重。
- [ ] 保持数据集路径、输入尺寸、随机种子和验证配置不变。

## 任务 5：验证

- [ ] 运行 `tests/test_adaptive_defect_modules.py`。
- [ ] 构建原生 `yolo11n.yaml` 并构建新配置，比较输出形状和参数规模。
- [ ] 执行最小前向/反向连通性测试。
- [ ] 检查旧模块默认开关均为关闭状态。
- [ ] 记录需要同步到远程服务器的文件列表。

