# MCSR 分尺度结构化剪枝 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前 YOLO11n+MCSR 模型增加一个简单、可复现的 BN-γ 分尺度 DepGraph 结构化剪枝入口，在保护 P3、MCSR 路由器和 Detect 输出层的同时降低参数量与 FLOPs。

**Architecture:** 新增独立的 `tools/prune_mcsr.py`，不修改训练前向和损失。脚本按 YOLO 顶层层号把阶段输出卷积分为 P3、P4、P5 三组，为每组设置 0.10、0.20、0.30 的局部剪枝率；根据 BN-γ 选择低重要性通道，再直接请求 Torch-Pruning DependencyGraph 维护残差与 Concat 依赖。MCSR 的三个直接输入层（16、19、22）、路由器和整个检测头作为保护边界。剪枝结果保存为包含完整模型对象的 Ultralytics checkpoint，并输出剪枝前后参数量、MACs 和前向形状。

**Tech Stack:** Python 3.11/3.12、PyTorch、Ultralytics 8.4.94、torch-pruning 1.6.0、pytest。

## Global Constraints

- 直接修改 `D:\个人文件整理\02_开发项目\ultralytics-8.4.94`，不创建 Git 分支、不提交、不推送。
- 不修改 MCSR、SA-TAL 的数学逻辑，不剪第 16/19/22 层 MCSR 输入接口、MCSR 路由器和 Detect 检测头内部输出通道。
- 默认剪枝率固定为 P3=0.10、P4=0.20、P5=0.30；通道数按 8 对齐；第一版只执行一次剪枝。
- P3 包括浅层骨干和 P3 Neck，优先保护 Surface_Crack 的细粒度特征。
- 所有生产代码遵循测试先行；测试必须先因功能缺失而失败，再实现最小代码使其通过。

---

### Task 1: 分尺度候选选择与保护规则

**Files:**
- Create: `tests/test_mcsr_pruning.py`
- Create: `tools/prune_mcsr.py`

**Interfaces:**
- Produces: `PruningConfig`、`stage_for_layer(int) -> str`、`collect_prunable_convs(nn.Module, PruningConfig) -> dict[nn.Conv2d, float]`。
- Consumes: Ultralytics `DetectionModel.model` 的顶层层序号和 `AdaptiveDefectDetect` 类型。

- [ ] **Step 1: 写失败测试**

```python
def test_stage_mapping_prioritizes_p3_protection():
    assert stage_for_layer(16) == "p3"
    assert stage_for_layer(19) == "p4"
    assert stage_for_layer(22) == "p5"


def test_collect_prunable_convs_excludes_detection_head_and_router():
    model = build_mcsr_model()
    ratios = collect_prunable_convs(model.model, PruningConfig())
    names = {name for name, module in model.model.named_modules() if module in ratios}
    assert names
    assert all(not name.startswith("23") for name in names)
    assert all("router" not in name for name in names)
```

- [ ] **Step 2: 运行失败测试**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_mcsr_pruning.py -q`

Expected: FAIL，原因是 `tools.prune_mcsr` 尚不存在。

- [ ] **Step 3: 实现最小选择逻辑**

```python
@dataclass(frozen=True)
class PruningConfig:
    p3_ratio: float = 0.10
    p4_ratio: float = 0.20
    p5_ratio: float = 0.30
    round_to: int = 8


def stage_for_layer(index: int) -> str:
    if index in {0, 1, 2, 3, 4, 14, 15, 16, 17}:
        return "p3"
    if index in {5, 6, 11, 12, 13, 18, 19, 20}:
        return "p4"
    if index in {7, 8, 9, 10, 21, 22}:
        return "p5"
    return "protected"
```

`collect_prunable_convs` 仅收集 0--22 层内的 `nn.Conv2d`；第 23 层 `AdaptiveDefectDetect` 整体排除。

- [ ] **Step 4: 运行测试确认通过**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_mcsr_pruning.py -q`

Expected: PASS。

### Task 2: BN-γ DepGraph 剪枝执行器

**Files:**
- Modify: `tools/prune_mcsr.py`
- Modify: `tests/test_mcsr_pruning.py`

**Interfaces:**
- Consumes: `collect_prunable_convs` 返回的逐层剪枝率。
- Produces: `prune_model(model, example_inputs, config) -> PruningReport`。

- [ ] **Step 1: 写失败测试**

```python
def test_prune_model_reduces_parameters_and_keeps_forward_valid():
    model = build_mcsr_model().model.eval()
    sample = torch.zeros(1, 3, 160, 160)
    before = sum(p.numel() for p in model.parameters())
    report = prune_model(model, sample, PruningConfig())
    after = sum(p.numel() for p in model.parameters())
    output = model(sample)
    assert report.after_params == after < before
    assert output is not None
```

- [ ] **Step 2: 运行测试确认因函数缺失而失败**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_mcsr_pruning.py::test_prune_model_reduces_parameters_and_keeps_forward_valid -q`

Expected: FAIL，原因是 `prune_model` 尚未实现。

- [ ] **Step 3: 实现单步剪枝**

对每个候选阶段输出卷积，根据配对 BatchNorm 的 `abs(gamma)` 升序选择通道，并将保留宽度向上对齐到 8。每个候选根单独通过 `tp.DependencyGraph.get_pruning_group()` 获取依赖组并执行；不使用会枚举全模型卷积根、在 Windows 上初始化过重的 `MagnitudePruner`。第 16/19/22 层保持固定宽度，避免通用图跟踪把 MCSR 的 mean/std 描述子错误解释为通道依赖。剪枝前后分别调用 `tp.utils.count_ops_and_params`，若参数量没有下降或前向失败则抛出明确异常。

- [ ] **Step 4: 运行测试确认通过**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_mcsr_pruning.py -q`

Expected: PASS，参数量下降且 160×160 前向可运行。

### Task 3: checkpoint 保存和命令行入口

**Files:**
- Modify: `tools/prune_mcsr.py`
- Modify: `tests/test_mcsr_pruning.py`

**Interfaces:**
- Produces: `save_pruned_checkpoint(yolo, output_path, report)`，命令行参数 `--weights`、`--output`、`--imgsz`、`--p3-ratio`、`--p4-ratio`、`--p5-ratio`。
- Consumes: 已训练的 MCSR checkpoint；未提供时默认使用 `yolo11n.pt` 迁移构建 MCSR 模型，作为连通性测试而非最终论文权重。

- [ ] **Step 1: 写失败测试**

```python
def test_pruned_checkpoint_can_be_reloaded(tmp_path):
    yolo = build_mcsr_model()
    report = prune_model(yolo.model, torch.zeros(1, 3, 160, 160), PruningConfig())
    path = save_pruned_checkpoint(yolo, tmp_path / "mcsr_pruned.pt", report)
    restored = YOLO(path)
    assert sum(p.numel() for p in restored.model.parameters()) == report.after_params
    assert restored.model(torch.zeros(1, 3, 160, 160)) is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_mcsr_pruning.py::test_pruned_checkpoint_can_be_reloaded -q`

Expected: FAIL，原因是保存接口缺失。

- [ ] **Step 3: 实现保存和入口**

保存前把剪枝报告写入 checkpoint 的 `pruning` 元数据，并调用 Ultralytics 的 `YOLO.save()` 保存完整模型对象；输出目录自动创建。入口打印剪枝前后 Params、MACs、压缩比例以及输出路径。

- [ ] **Step 4: 运行定向测试和真实冒烟命令**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_mcsr_pruning.py -q`

Run: `D:\anaconda\envs\test\python.exe tools/prune_mcsr.py --imgsz 160 --output runs/prune/mcsr_scale_pruned_smoke.pt`

Expected: 测试全部 PASS；冒烟命令退出码 0，生成可重新加载的 `.pt`，参数量和 MACs 均低于剪枝前。
