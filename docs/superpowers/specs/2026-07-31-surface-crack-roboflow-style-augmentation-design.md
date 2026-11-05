# Surface Crack Roboflow 风格增强设计

## 目标

参照 `datasets（15种类别）` 中可观察到的 Roboflow 导出变体，将适合细裂缝目标检测的几何、颜色和轻度成像扰动应用到当前 Surface Crack 数据集，同时避免训练集与验证集发生同源泄漏。

## 已确认事实

- 15 类数据集没有发现明确的增强配置文件，因此不能声称复现了其精确参数。
- 同一基础文件名存在多个 `.rf.<hash>` 变体；抽样图像显示出旋转、边界填充、模糊/噪声和颜色差异。
- 当前 Surface Crack 数据集中的裂缝较细，强裁剪、大幅模糊和 Mosaic 可能破坏目标信息。

## 设计

- 以当前质量筛选后的原始图像为输入，保持现有数据不覆盖。
- 训练集：保留原图，并生成水平翻转、轻度仿射、亮度/对比度扰动、轻度噪声/模糊和轻度旋转组合样本。
- 验证集：只保留原始图片，不进行离线增强，保证 mAP 与基线评价口径一致。
- 所有几何变换通过 YOLO 格式边界框同步变换；坐标裁剪到 `[0, 1]`，删除无效框。
- 旋转使用反射边界填充，避免出现 15 类样例中的大块黑色三角边界。
- 不使用强裁剪、Mosaic、CutMix 和大幅模糊。

## 输出

输出目录：

`C:\Users\23223\Desktop\CoatingVision_Surface_Crack_augmented_v2_augment_first\roboflow_style_augmented`

包含：

- `images/train`、`labels/train`
- `images/val`、`labels/val`
- `data.yaml`：训练增强集 + 原始验证集
- `generation_report.json`：数量、来源、增强类型和质量检查结果

## 验证

- 每张图片都有对应标签，且数量一致。
- 标签类别全部为 `0`，YOLO 框值位于合法范围。
- 训练集与验证集按源文件 ID 检查无交集。
- 使用 Python 编译检查和独立统计脚本验证输出。
