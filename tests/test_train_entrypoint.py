# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import importlib
import inspect


def test_train_exposes_cli_and_function_defaults_consistently():
    """The CLI must expose the same documented defaults as the callable training entry point."""
    module = importlib.import_module("train")
    parameters = inspect.signature(module.train_model).parameters
    source = inspect.getsource(module)

    assert hasattr(module, "parse_args")
    # epochs 采用 None->DEFAULT_EPOCHS 单一事实来源（函数默认不重复写死，避免与文件常量漂移）
    assert parameters["epochs"].default is None
    assert module.DEFAULT_EPOCHS == 5
    assert parameters["imgsz"].default == 640
    # 已证伪/重叠模块的开关应已全部移除（mcsr/shde/scol/fcfl）
    assert "mcsr" not in parameters
    assert "shde" not in parameters
    assert "scol" not in parameters
    assert "fcfl" not in parameters
    # 已删除证伪特性：SA-TAL / AR-Prior（50 轮裁决后，见上下文快照 0e）
    assert "sa_tal" not in parameters
    assert "sa_size" not in parameters
    assert "ar_prior" not in parameters
    assert "ar_min" not in parameters
    assert "ar_max" not in parameters
    assert "ar_radius" not in parameters
    # 当前唯一保留特性：consensus 推理重排
    assert parameters["consensus_rerank"].default is False
    # 网络侧只有 stock yolo11.yaml（MCSR/AdaptiveDefectDetect 已被证伪删除）
    assert "yolo11n-adaptive-defect.yaml" not in source
    assert "yolo11.yaml" in source


def test_train_short_experiment_uses_one_epoch_warmup_by_default():
    module = importlib.import_module("train")
    parameters = inspect.signature(module.train_model).parameters

    assert parameters["warmup_epochs"].default == 1.0
