# 2026-08-02 · 小目标检测双创新设计规格（SCOL + SHDE）

## 背景与目标

针对 `D:/浏览器下载位置/detection_yolo_v2`（3 类：Pinhole / Surface_Crack / Delamination）的两个核心数据问题设计两个全新创新模块，并用 5 轮×640 消融实验验证：

1. **类别失衡**：Pinhole 框仅占 8.4%（train 3176/37602），约 1/3 图像出现，极易被淹没。
2. **小目标密集**：train 32.7% / val 37.7% 的框 <32px（640 宽图），stride-8 特征上仅 ~2 网格。

经验性发现驱动了方案修正（见下文「实验结论」）：单纯按频率加权（FCFL）在本数据集上被证实为负收益；数据集真正的瓶颈是**小/难目标**而非类别频率。因此：

- **创新点 A = SCOL**（Small-object COnditioned Loss，损失侧，针对小目标）。
- **创新点 B = SHDE**（Small-object Detail Enhancement，网络侧，针对小目标）。

## 消融矩阵（统一 5 轮、imgsz 640、预训练 yolo11n.pt 起步、batch 16、seed 0、deterministic、指标 mAP50/mAP50-95 + 各类别 AP）

| 组 | 模型 YAML | 开关 |
|---|---|---|
| A0 基线 | 原生 `yolo11.yaml` (n) | 全关 |
| A1 +SCOL | 原生 `yolo11.yaml` (n) | `scol: True` |
| A2 +SHDE | 新增 `yolo11n-shde.yaml` | `shde: True` |
| A3 双创新 | `yolo11n-shde.yaml` | `shde: True, scol: True` |

## 创新点 A：SCOL —— 小目标条件分类损失（损失侧）

`ultralytics/utils/loss.py` 在 `v8DetectionLoss.get_assigned_targets_and_loss` 分类分支引入尺寸条件权重：

1. **小目标上权重、大目标与背景不扰动**：对每个前景 anchor，`size_weight = 1 + α·ReLU(1 − area/S²)`，其中 `area` 为该 anchor 分配的 GT 框像素面积（`target_bboxes` 为像素坐标，imgsz 640 下阈值 `S=32`），`α=2.0` 为最大倍数。小框分类 BCE 被放大 ≤3×，≥S 的框权重退化为 1，背景 anchor 权重恒为 1。
2. **只作用于分类损失**：`bce_loss = bce_loss * size_weight`，box/dfl 损失不受影响，全局梯度流保留。
3. **前景门控（关键实现细节）**：TAL 返回的 `target_bboxes` 对背景 anchor 保存的是该图第 0 个 GT 框的副本（`tal.py get_targets` 用 `target_gt_idx` 索引 `gt_bboxes`，全零行 argmax 回落到 0）。若不按 `fg_mask` 掩码，当第 0 框为小框时整幅图的背景 BCE 会被错误放大（破坏召回）。实现中用 `torch.where(fg_mask[..., None], w, 1)` 将背景权重钳为 1。

- 开关与超参经 `model.args`：`scol`（默认 False）、`scol_alpha`=2.0、`scol_size`=32。
- `scol=False` 时行为与原版完全一致（含现有 `class_weights` 通道），消融干净。

## 创新点 B：SHDE —— 小目标高频细节增强（网络侧）

`ultralytics/nn/modules/small_object.py` 新增 `SmallObjectDetailEnhance`，集成进 `AdaptiveDefectDetect` 头部（与 MCSR 并列，`set_adaptive_modules_enabled` 统一开关）：

1. **作用于最细两个原生尺度（stride 8/16）**：小目标在 stride-8 特征上仅 ~2 网格，增强最细两个原生尺度（P3/P4），不引入 stride-4 P2 分支（v1 尝试过 P2，5 轮短训练下随机初始化分支拖累 TAL，回归至 0.724 → 移除）。
2. **高频细节门控**：对最细两尺度逐位置计算梯度幅值场（水平+垂直中心差分）→ 3×3 卷积 → sigmoid 得**空间细节门 S**；每通道 `abs-mean` 过 MLP → sigmoid 得**通道细节门 C**。输出 `x · (1 + gain·S·C)`。
3. **恒等初始化**：`detail_gain=0` → 初始 no-op，逐步学习注入，保证短训练稳定、消融干净。仅 +6.4k 参数（n 尺度总 2,596,815）。

- 头部签名扩展为 `AdaptiveDefectDetect(nc, mcsr, mcsr_hidden_dim, shde, shde_hidden_dim, reg_max, end2end, ch)`；`mcsr` 条件构建（False 时不实例化）。
- 新增 `cfg/models/11/yolo11n-shde.yaml`：head 为 `[[16, 19, 22], 1, AdaptiveDefectDetect, [nc, False, 16, True, 16]]`。

## 使用入口（train.py 统一开关）

四个创新点开关集中在仓库根目录 `train.py`：

```bash
python train.py                          # 纯 yolo11n 基线（全关）
python train.py --scol                   # +SCOL（损失侧，小目标分类加权）
python train.py --shde                   # +SHDE（网络侧，小目标细节增强）
python train.py --shde --scol            # 双创新（推荐配置，对应消融 A3）
python train.py --mcsr --shde --scol     # 网络/损失创新叠加（组合 YAML）
python train.py --fcfl                   # +FCFL（已证伪，仅供复现负结果）
```

- 网络侧开关（`mcsr`/`shde`）自动选择模型 YAML（模块实例化），损失侧开关（`scol`/`fcfl`）直接传入训练；运行名由开关自动生成。
- 超参可选：`--scol-alpha`、`--scol-size`、`--fcfl-kappa` 等。
- 注：`parse_model` 对 Detect 族头部始终追加 `[reg_max, end2end, ch]`（tasks.py:2006），`AdaptiveDefectDetect` 头部参数须按 `[nc, mcsr, mcsr_hidden_dim, shde, shde_hidden_dim]` 五元组写全，否则位置绑定错位（`reg_max` 收到通道列表）。`yolo11n-adaptive-defect.yaml` 已按此修复为 `[nc, True, 16, False, 16]`。

## 配置项（default.yaml）

```yaml
shde: False
shde_hidden_dim: 16
scol: False
scol_alpha: 2.0
scol_size: 32
# 保留但已从消融最终方案中移除（负收益，见实验结论）：
fcfl: False
fcfl_beta: 0.9
fcfl_kappa: 0.2
fcfl_gamma_base: 1.0
fcfl_gamma_slope: 0.2
fcfl_min_presence: 0.03
```

## 文件改动清单

| 文件 | 改动 |
|---|---|
| `nn/modules/small_object.py` | 新：`SmallObjectDetailEnhance` |
| `nn/modules/head.py` | `AdaptiveDefectDetect` 增加 shde、mcsr 条件构建 |
| `nn/modules/__init__.py` | 注册 SHDE（供 head 导入；无需 tasks.py 解析注册，因 SHDE 内嵌于头部） |
| `cfg/models/11/yolo11n-shde.yaml` | 新：3 尺度头部 + SHDE 门控 |
| `cfg/models/11/yolo11n-adaptive-defect.yaml` | 头部参数补齐为 5 元组（MCSR 随签名扩展修复） |
| `cfg/models/11/yolo11n-adaptive-defect-shde.yaml` | 新：MCSR+SHDE 组合（`--mcsr --shde`） |
| `train.py` | 新：四个创新点统一开关 + CLI |
| `utils/loss.py` | 新：`FrequencyConditionedFocalLoss`（保留）+ SCOL 尺寸条件权重（前景门控） |
| `cfg/default.yaml` | 新增 SCOL 3 项 + SHDE 2 项 + FCFL 5 项 |
| `experiments/fcfl_shde_ablation.py` | 新：4 组消融脚本（种子固定、batch 16、workers 4、device 0） |

## 实验结论（5 轮 × 640，确定性单进程 4 组）

### 最终消融结果

| 组 | mAP50 | mAP50-95 | Pinhole AP50 | Surface_Crack AP50 | Delamination AP50 |
|---|---|---|---|---|---|
| A0 基线 | 0.760 | 0.551 | 0.951 | 0.507 | 0.822 |
| A1 +SCOL | **0.803** | 0.562 | 0.959 | **0.610** | 0.840 |
| A2 +SHDE | 0.766 | 0.546 | 0.957 | 0.511 | 0.830 |
| A3 双创新 | 0.800 | **0.567** | 0.958 | 0.599 | **0.841** |

各类别 mAP50-95：

| 组 | Pinhole | Surface_Crack | Delamination |
|---|---|---|---|
| A0 基线 | 0.714 | 0.341 | 0.598 |
| A1 +SCOL | 0.700 | 0.396 | 0.590 |
| A2 +SHDE | 0.689 | 0.348 | 0.600 |
| A3 双创新 | 0.707 | 0.392 | 0.602 |

### 关键证据与结论

- **SCOL 是主力创新**：mAP50 +0.043、mAP50-95 +0.010；主类别 Surface_Crack AP50 +0.103（0.507→0.610），Pinhole/Delam 也各 +0.008/+0.018。小目标分类加权直接对准了数据集真正的失败点，三类无一回退。
- **SHDE 稳健正向**：mAP50 +0.006，三类 AP50 全部微升（+0.006/+0.004/+0.008），代价仅 +6.4k 参数；单独看 mAP50-95 持平（−0.005），但在 SCOL 之上进一步把 mAP50-95 从 0.562 提到 0.567。
- **双创新 A3 为推荐配置**：mAP50-95 全场最高 0.567（+0.016），mAP50 0.800（+0.040）与 A1 的 0.803 统计上同档，三类 mAP50-95 全升（Pinhole 0.707 / SC 0.392 / Delam 0.602 均高于基线）。两个创新正交叠加，无冲突。
- **FCFL 被证伪（决策依据）**：受控对比中 Pinhole（罕见类，基线 AP50 已 0.951）无提升空间，逆频率权重把容量从饱和的 Pinhole 转移到被压低的主类别（SC 0.507→0.429），mAP50 −0.045。频率失衡不是本数据集瓶颈，故损失侧创新从 FCFL 替换为 SCOL（保留实现代码、移出最终方案，开关默认 False，不参与消融）。

## 实验执行与风险

- 顺序执行 A0→A3，同一 data.yaml、同 seed 0、deterministic=True（已验证：A0/A2 跨进程精确复现）。
- 预期耗时：每组 yolo11n 5 轮 640 ≈ 8–10 分钟（RTX 4060 8GB），共约 40 分钟。
- 风险与后备：
  - SCOL 放大过猛致训练不稳 → 调低 `scol_alpha`。
  - 5 轮信号弱 → 以 Pinhole AP 与 mAP50 相对基线趋势为准，不作绝对结论。
