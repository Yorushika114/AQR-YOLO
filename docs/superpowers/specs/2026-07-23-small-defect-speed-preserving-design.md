# 电池表面小缺陷精度与速度平衡设计

## 目标

在保持现有 MDRH、BCRH、CDAA 三个创新点定义和独立开关不变的前提下，针对当前数据集中针孔和细裂纹占比较高、目标尺寸很小的问题，提升 `mAP50-95`。当前实验结果为三个模块全开约 0.534，基线约 0.527；0.58–0.60 作为复训目标区间，不能在复训前视为保证结果。

## 数据证据

当前 `detection_yolo_v2` 训练集包含 62,996 个目标，验证集包含 15,560 个目标。验证集中的 `Surface_Crack` 有 11,493 个实例，`Pinhole` 有 776 个实例；按标注框归一化面积统计，针孔中约 88% 小于图像面积的 0.25%，细裂纹中约 56% 小于该面积。当前 `train.py` 默认 `imgsz=160`，基础 `yolo11.yaml` 只输出 P3/P4/P5，因此细小目标在进入检测头前已经面临明显的空间信息损失。

## 设计原则

1. 不新增完整 P2 检测头、Transformer、大型注意力或额外完整前向分支，避免再次出现数分钟一轮的训练速度。
2. MDRH 仍然负责点状、线状、区域形态的动态路由；BCRH 仍然负责中心/边界响应辅助监督；CDAA 仍然负责类别密度自适应任务对齐分配。
3. 新增逻辑只在现有三模块内部增强：局部细节对比用于 MDRH 路由，非零边界调制和边界加权用于 BCRH，目标尺寸—检测层步长匹配用于 CDAA。
4. 基线开关全部关闭时，新增逻辑不参与计算，保持原始 YOLO11 路径。

## 模块设计

### 1. MDRH：局部对比度细节引导形态路由

对每个已有检测尺度的输入特征 `x` 计算无参数局部高频响应，并再减去其局部背景：

```text
detail = x - AvgPool(x, kernel=3, stride=1, padding=1)
raw_detail = mean(abs(detail), channel)
detail_energy = ReLU(raw_detail - AvgPool(raw_detail, kernel=5, stride=1, padding=2))
```

将归一化后的 `detail_energy` 作为已有 point/line/region 路由的局部偏置，并提高 line 分支的细节偏置。它不改变三条形态分支，不增加卷积层，只在已有路由 logits 上进行逐位置调节。局部对比度主要帮助线状裂纹边缘和点状针孔从反光或纹理背景中分离出来；区域分支仍由全局路由和原有区域响应控制。

### 2. BCRH：尺度感知中心/边界辅助监督

保留逐尺度门控和 Soft Dice 损失，为每个输出尺度增加显式权重。高分辨率 P3 的权重略高，低分辨率 P5 的权重略低，默认使用 `[1.25, 1.0, 0.75]`，并在损失内部归一化。BCRH 保持中心/边界双响应，边界通道使用 `1.5` 的监督权重，同时使用最大增益为 `0.10` 的非零边界调制门，使边界形态证据直接参与检测特征。轴线响应实验已撤回，因为它在当前数据集上造成类别性能退化。

### 3. CDAA：类别密度与目标尺度联合分配

在已有类别稀有度、局部密度、长宽比一致性和面积补偿基础上，使用每个候选位置的检测层步长进行尺度匹配。目标的期望步长约为目标尺度的四分之一，匹配质量乘到任务对齐度量中，使小裂纹更倾向于 P3，高尺度目标仍可匹配 P4/P5：

```text
small_factor = clip((median_area / gt_area) ** 0.20, 1.0, 1.75)
adaptive_topk = clip(class_density_topk * small_factor, min_topk, max_topk)
scale_quality = exp(-abs(log(anchor_stride / preferred_stride)))
align_metric = align_metric * (1 + 0.35 * scale_quality)
```

该计算只作用于任务对齐分配阶段，不增加模型前向分支。对异常极小框仍使用 top-k 上限约束，避免正样本数量失控。

## 配置

新增：

```yaml
mdrh_detail_weight: 0.15
bcrh_level_weights: [1.25, 1.0, 0.75]
bcrh_weight: 0.08
bcrh_boundary_weight: 1.5
bcrh_gate_gain: 0.10
cdaa_small_object_power: 0.20
cdaa_scale_weight: 0.35
```

训练时仍使用：

```text
mdrh=True, bcrh=True, cdaa=True
```

所有对照实验继续固定数据划分、随机种子、`imgsz`、batch、epoch、优化器和验证脚本。建议第一轮仍使用 `imgsz=160` 验证模块收益；若收益不足，再单独比较 `imgsz=192`，不能同时改变输入尺寸和模块结构。

## 验证标准

- 新增测试覆盖高频输入能够改变 MDRH 路由、BCRH 多尺度权重归一化、CDAA 小目标 top-k 上限和独立开关。
- 运行模块测试、验证脚本测试、模型前向/反向连通性测试。
- 固定尺寸 CPU 微基准中，新增 MDRH 路由和 CDAA 分配开销不应出现数量级增长。
- 远程复训至少记录整体及三类的 Precision、Recall、mAP50、mAP50-95、每轮耗时和最佳轮次；只有同条件下验证结果达到目标并且训练速度未显著退化，才能认为方案有效。
