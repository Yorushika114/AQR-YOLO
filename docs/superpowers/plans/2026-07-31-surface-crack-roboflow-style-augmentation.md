# Surface Crack Roboflow Style Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 Surface Crack 数据集内生成带同步 YOLO 标注的 Roboflow 风格增强集，并保留原始与增强验证集以便公平比较。

**Architecture:** 新增一个独立的数据准备脚本，读取质量筛选后的原始样本，使用 Albumentations 和手工仿射框变换完成图像和边界框同步变换，将训练增强集和原始验证集写入新目录；原目录和既有输出不修改。

**Tech Stack:** Python 3、Pillow、OpenCV、Albumentations、JSON、YOLO 标签格式。

## Global Constraints

- 不删除或覆盖现有数据。
- 只对训练集生成多份增强样本；验证集原图必须保留。
- 任何几何变换必须同步变换边界框。
- 不使用强裁剪、Mosaic、CutMix 或大幅模糊。

---

### Task 1: 建立失败测试

**Files:**
- Create: `tests/test_surface_crack_roboflow_augmentation.py`
- Create: `tools/apply_roboflow_style_surface_aug.py`

**Interfaces:**
- `build_pipeline(kind: str, seed: int)` 返回可调用的 Albumentations 管线。
- `validate_yolo_label_file(path: Path)` 返回错误列表。
- `build_dataset(...)` 生成输出目录和报告。

- [ ] **Step 1: 写测试，先验证输出目录和标签校验接口尚不存在**

```python
from pathlib import Path

from tools.apply_roboflow_style_surface_aug import validate_yolo_label_file


def test_invalid_yolo_box_is_rejected(tmp_path: Path):
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5 1.2 0.2\n", encoding="utf-8")
    assert validate_yolo_label_file(label)
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

运行：`D:\anaconda\envs\test\python.exe -m pytest tests/test_surface_crack_roboflow_augmentation.py -q`

预期：FAIL，原因是待实现模块不存在。

### Task 2: 实现最小标签校验和增强管线

- [ ] **Step 1: 实现 `validate_yolo_label_file` 和 `build_pipeline`**
- [ ] **Step 2: 重新运行单元测试，确认非法框被识别**
- [ ] **Step 3: 添加合法标签、水平翻转和框同步变换测试**
- [ ] **Step 4: 运行测试，确认变换后的框仍在合法范围内**

### Task 3: 实现数据集生成

- [ ] **Step 1: 从 `quality_filtered_split` 读取原图和标签**
- [ ] **Step 2: 生成训练原图与增强图**
- [ ] **Step 3: 生成只包含原图的验证集**
- [ ] **Step 4: 写入 `data.yaml` 和 `generation_report.json`**

### Task 4: 独立验证输出

- [ ] **Step 1: 运行生成脚本**
- [ ] **Step 2: 统计图片、标签、类别、框数量**
- [ ] **Step 3: 检查训练/验证源 ID 交集**
- [ ] **Step 4: 编译脚本并运行测试**
