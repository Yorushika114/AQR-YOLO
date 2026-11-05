# Pruned MCSR Recovery Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个独立入口，在不按 YAML 重建模型的前提下，对 MCSR 结构化剪枝 checkpoint 进行恢复微调并复核最终权重结构。

**Architecture:** `tools/finetune_pruned_mcsr.py` 负责输入审计、训练参数构建、`DetectionTrainer` 注入和训练后复核。训练时把 checkpoint 中的真实剪枝 `nn.Module` 直接赋给训练器，使 `BaseTrainer.setup_model()` 跳过 `get_model()`；训练结束后重新加载 `best.pt`/`last.pt`，校验参数量和 MCSR 路由器，并补写来源剪枝元数据。

**Tech Stack:** Python、PyTorch、Ultralytics 8.4.94、pytest。

## Global Constraints

- 只修改 `D:\个人文件整理\02_开发项目\ultralytics-8.4.94` 本地文件，不创建分支、不提交、不推送。
- 不修改普通 `train.py`、MCSR 数学逻辑、SA-TAL 数学逻辑和现有剪枝比例。
- 微调必须保留 checkpoint 中的真实剪枝通道，微调后参数量必须与微调前完全相同。
- 第一版只支持单进程单设备训练，拒绝会重新启动进程并丢失注入模型的多 GPU DDP 配置。
- 默认启用 SA-TAL；MCSR 必须作为结构存在并参与梯度更新。
- 所有生产代码遵循测试先行；不使用 Git 操作。

---

### Task 1: 剪枝 checkpoint 审计

**Files:**
- Create: `tests/test_pruned_mcsr_finetune.py`
- Create: `tools/finetune_pruned_mcsr.py`

**Interfaces:**
- Produces: `PrunedModelAudit`、`load_and_audit_pruned_model(weights: str | Path) -> tuple[YOLO, PrunedModelAudit]`。
- Consumes: checkpoint 顶层 `pruning.before_params`、`pruning.after_params` 和带 `router` 的检测头。

- [ ] **Step 1: Write the failing checkpoint-audit tests**

测试构造一个带合法剪枝元数据的本地 MCSR checkpoint，并断言审计返回实际参数量；另用 `yolo11n.pt` 断言抛出包含 `pruning metadata` 的 `ValueError`。

- [ ] **Step 2: Run tests to verify RED**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: FAIL during collection because `tools.finetune_pruned_mcsr` does not exist.

- [ ] **Step 3: Implement the minimal audit API**

`PrunedModelAudit` 保存 `source`、`before_params`、`after_params`、`current_params` 和 `class_count`。`load_and_audit_pruned_model()` 使用 `YOLO(weights)` 加载完整模型，要求权重存在、`pruning` 为字典、`before_params > after_params == current_params`，并要求最后一层 `router` 非空。

- [ ] **Step 4: Run tests to verify GREEN**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: checkpoint audit tests PASS.

### Task 2: 结构保持型训练器注入与参数构建

**Files:**
- Modify: `tests/test_pruned_mcsr_finetune.py`
- Modify: `tools/finetune_pruned_mcsr.py`

**Interfaces:**
- Consumes: `YOLO` 和 `PrunedModelAudit`。
- Produces: `FineTuneConfig`、`build_train_overrides(config) -> dict`、`build_structure_preserving_trainer(yolo, overrides, trainer_cls=DetectionTrainer)`。

- [ ] **Step 1: Write failing injection and configuration tests**

断言默认 overrides 包含 `lr0=0.001`、`optimizer='AdamW'`、`sa_tal=True`、`sa_size=40.0`、`resume=False` 和 `pretrained=False`；断言多设备列表或逗号分隔设备被拒绝；断言注入后 `trainer.model is yolo.model`，并且调用 `trainer.setup_model()` 不触发替代模型构建。

- [ ] **Step 2: Run tests to verify RED**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: FAIL because the configuration and trainer APIs are missing.

- [ ] **Step 3: Implement configuration and injection**

`FineTuneConfig` 提供可直接编辑的权重、数据集、epochs、batch、workers、imgsz、device、project、name、lr0、lrf、warmup_epochs、close_mosaic、SA-TAL 参数。`build_structure_preserving_trainer()` 实例化 `DetectionTrainer` 后直接设置 `trainer.model = yolo.model`，同时保持回调；不调用 `YOLO.train()`。

- [ ] **Step 4: Run tests to verify GREEN**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: configuration and injection tests PASS.

### Task 3: 微调执行、训练后复核与元数据

**Files:**
- Modify: `tests/test_pruned_mcsr_finetune.py`
- Modify: `tools/finetune_pruned_mcsr.py`

**Interfaces:**
- Consumes: `FineTuneConfig`、`PrunedModelAudit` 和训练器生成的 `best.pt`/`last.pt`。
- Produces: `FineTuneResult`、`verify_finetuned_checkpoint(path, expected_params) -> int`、`annotate_finetuned_checkpoint(path, audit, config)`、`finetune_pruned_model(config) -> FineTuneResult`。

- [ ] **Step 1: Write failing post-training verification tests**

断言相同结构 checkpoint 复核成功、参数量不同 checkpoint 被拒绝、MCSR 路由器缺失被拒绝；断言补写后 checkpoint 包含 `pruning` 和 `pruning_finetune`，并仍可通过 `YOLO(path)` 加载。

- [ ] **Step 2: Run tests to verify RED**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: FAIL because final-checkpoint verification is missing.

- [ ] **Step 3: Implement execution and final audit**

`finetune_pruned_model()` 依次执行路径检查、剪枝审计、训练器注入、`trainer.train()`、权重选择、参数量复核和元数据补写。补写元数据时保留 checkpoint 原有模型、指标与训练参数，只新增来源、微调前后参数量和 SA-TAL 配置。

- [ ] **Step 4: Run tests to verify GREEN**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: all fine-tuning tests PASS.

### Task 4: 命令入口与完整验证

**Files:**
- Modify: `tools/finetune_pruned_mcsr.py`
- Modify: `tests/test_pruned_mcsr_finetune.py`

**Interfaces:**
- Produces: `parse_args()` 和 `main()`；脚本顶部默认路径可直接修改，命令行参数可覆盖。

- [ ] **Step 1: Write failing CLI-default tests**

断言默认配置指向 `runs/prune/mcsr_scale_pruned.pt` 和 `D:\浏览器下载位置\detection_yolo_v2\data.yaml`，并检查帮助文本包含 `--weights`、`--data`、`--epochs`、`--sa-tal` 和 `--no-sa-tal`。

- [ ] **Step 2: Run test to verify RED**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: FAIL because CLI wiring is incomplete.

- [ ] **Step 3: Implement CLI and concise result output**

`main()` 打印来源权重、微调前参数量、输出目录、最终权重和微调后参数量。直接运行脚本使用顶部默认值；命令行传值覆盖默认值。

- [ ] **Step 4: Run focused and regression verification**

Run: `D:\anaconda\envs\test\python.exe -m pytest tests/test_pruned_mcsr_finetune.py tests/test_mcsr_pruning.py -q --basetemp .pytest_tmp_pruned_finetune`

Expected: all tests PASS.

Run: `D:\anaconda\envs\test\python.exe -m py_compile tools/finetune_pruned_mcsr.py tests/test_pruned_mcsr_finetune.py`

Expected: exit code 0.

- [ ] **Step 5: Run a no-training structure smoke check**

加载 `runs/prune/mcsr_scale_pruned_640_smoke.pt`，构建训练器但不启动 epoch，调用审计与 `setup_model()`，确认参数量保持为剪枝报告的 `after_params` 且 MCSR 路由器存在。
