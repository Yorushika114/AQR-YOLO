# CDAA 形态门控设计

## 目标

在保持 MDRH、BCRH 和现有 CDAA 结构不变的前提下，降低 Surface_Crack 训练早期因预测框长宽比不准而被错误抑制的风险，重点改善漏检和定位质量。

## 现状与根因

当前 CDAA 在 `TaskAlignedAssigner.get_box_metrics` 中直接把预测框与真实框的长宽比一致性乘到 alignment metric 上。训练早期预测框 IoU 较低时，形态预测本来就不可靠，此时继续使用形态惩罚可能把本应保留的裂纹候选筛掉。最新验证中 Surface_Crack 的 Precision 高于 Recall，说明当前更需要保护候选召回，而不是继续提高抑制强度。

## 方案

新增 `cdaa_shape_iou_gate` 参数。仅对 `cdaa_focus_class`（当前为 Surface_Crack，类别 1）应用门控形态先验：

```text
gate = clamp((IoU - cdaa_shape_iou_gate) / (1 - cdaa_shape_iou_gate), 0, 1)
shape_factor = exp(-gate * |log(aspect_gt) - log(aspect_pred)|)
```

当 IoU 低于门限时，焦点类别仍保留 `cdaa_shape_floor` 比例的长宽比约束；当 IoU 逐渐提高时，形态一致性逐渐恢复到 100%。非焦点类别继续沿用原有 CDAA 形态权重，避免影响 Pinhole 和 Delamination 的已有行为。

## 约束

- 不新增特征图分支，不增加 MDRH/BCRH 推理分支。
- 只增加已有 IoU 张量上的逐元素运算，保持每轮训练速度基本不变。
- 默认门限为 `0.15`，作为显式配置写入训练参数。
- 默认形态约束下限为 `0.50`，作为显式配置写入训练和验证参数。
- 不保证单次 5 轮训练必然达到 0.58--0.60；最终判断以同数据、同随机种子、同验证脚本的消融结果为准。

## 验证

1. 单元测试：低 IoU 的 Surface_Crack 候选不因形态不一致而被额外压低；高 IoU 候选仍受形态一致性约束。
2. 回归测试：非焦点类别的形态评分保持原有逻辑。
3. 运行现有电池模块测试、验证脚本测试和 YAML/AST 检查。
4. 远端使用同一配置重新训练，比较 Surface_Crack Recall、mAP50-95 以及另外两类指标。
