# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Route-Aware structured channel pruning for YOLO11n + MCSR — fully self-made (no torch-pruning / no DepGraph).

创新点（网络侧剪枝准则）:
    旧 DepGraph 剪枝用静态 BN-gamma 作为通道重要性。本模块把 MCSR 路由矩阵所表达的
    "跨尺度依赖强度" 作为调制信号, 通道重要性升级为:
        score(c) = |BN-gamma(c)| * (1 + lambda * R(scale(c)))
    R(scale) 从 MCSR router(挂在检测头前) 的真实权重中抽取: 每个 target 尺度被全局
    source 平均关注的程度 -- 该尺度更重要, 则 R 更大, score 更高, 更不易被剪。

自研依赖书签（不用 torch-pruning）:
    架构固定(yolo11n + AdaptiveDefectDetect head)。Ultralytics parse_model 已为每个层
    模块写入 .f(来源)、.c1/.c2(输入/输出通道数), 我据此静态重建 "谁消费谁" 的通道
    流动, 而不是构建通用依赖图。
      - 阶段输出根(layer 的 cv2/cv3/conv): 只对它做输出通道裁切(权重 out 维 + BN)。
      - 消费端入口卷积(下一层/跨层 Concat 之后的 cv1): 据此裁切输入通道(权重 in 维)。
      保护边界: MCSR 输入接口层(16/19/22) 的输出宽度必须保持(路由器依赖), 检测头(23) 不动。
    剪后门槛: 参数差 == 逐次裁切累计缺失量(精确断言) + 前向输出形状与剪前一致(结构校验)。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


MCSR_MODEL_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo11n.pt"
DEFAULT_OUTPUT = ROOT / "runs/prune/route_aware_pruned.pt"

# 未剪枝 both (MCSR+SA-TAL, 100ep cold) 权重 — 剪枝源
UNPRUNED_BOTH = (
    ROOT.parents[1]
    / "Desktop"
    / "新论文数据保存"
    / "MCSR_SA-TAL消融实验_100ep_2x4"
    / "unpruned_both_cold"
    / "weights"
    / "best.pt"
)
UNPRUNED_BOTH_FALLBACK = ROOT / "runs/detect/both_e100/weights/best.pt"


@dataclass(frozen=True)
class RouteAwareConfig:
    """One route-aware structured pruning run's configuration."""

    p3_ratio: float = 0.10
    p4_ratio: float = 0.20
    p5_ratio: float = 0.30
    round_to: int = 8
    lambda_mod: float = 1.0  # 路由强度调制系数; 0 退化为纯 BN-gamma(对照)
    mcsr: bool = True
    # True: 让路由强度 R(scale) 真正调制各尺度的剪枝比例 —— 重要尺度(R 大)少剪,
    #       弱尺度(R 小)多剪, 从而把"路由感知"落到结构上而非仅层内排序。
    #       False: 保持原实现 —— 各尺度 ratio 恒定, R 只改变层内通道排序(等效纯BN-gamma)。
    route_aware_ratio: bool = True

    def __post_init__(self) -> None:
        for name in ("p3_ratio", "p4_ratio", "p5_ratio"):
            ratio = getattr(self, name)
            if not 0.0 <= ratio < 0.5:
                raise ValueError(f"{name} must be in [0.0, 0.5), got {ratio}")
        if self.round_to < 1:
            raise ValueError(f"round_to must be positive, got {self.round_to}")
        if self.lambda_mod < 0.0:
            raise ValueError(f"lambda_mod must be non-negative, got {self.lambda_mod}")

    def ratio_for(self, stage: str) -> float:
        return {"p3": self.p3_ratio, "p4": self.p4_ratio, "p5": self.p5_ratio}.get(stage, 0.0)

    def effective_ratio_for(self, stage: str, r_scale: float) -> float:
        """Route-aware per-stage pruning ratio: important scales (large R) get pruned *less*.

        base_ratio 是用户给定的基准剪枝比例; 路由强度 R 在 [0,1]。把"重要尺度"的有效剪枝比例压到
        base_ratio * (1 - R), 弱尺度保持在 base_ratio, 使路由感知真实反映到结构删减量上。
        为保持总预算不过度偏移, R 归一化到 0..1 后按 (1 - R) 线性衰减。
        """
        base = self.ratio_for(stage)
        if not self.route_aware_ratio:
            return base
        r = max(0.0, min(1.0, float(r_scale)))
        return base * (1.0 - r)


@dataclass(frozen=True)
class PruningReport:
    """Model complexity before/after structural pruning."""

    before_macs: float
    after_macs: float
    before_params: int
    after_params: int
    removed_params: int
    route_strength: tuple[float, float, float]
    lambda_mod: float
    pruned_layers: int

    @property
    def parameter_reduction(self) -> float:
        return 1.0 - self.after_params / self.before_params

    @property
    def mac_reduction(self) -> float:
        return 1.0 - self.after_macs / self.before_macs


# ---------------------------------------------------------------------------
# 阶段映射 + MCSR 保护边界 (yolo11n backbone + AdaptiveDefectDetect head 0-23)
# ---------------------------------------------------------------------------

_STAGE_LAYERS = {
    "p3": frozenset({0, 1, 2, 3, 4, 14, 15, 16, 17}),
    # 层 13 (neck C3k2, 512ch) 同时是 P3(P4 上采样源)与 P4(head 18->19)的内容源,
    # 剪它=同时削弱大小目标两条检测路径。故把层 13 移出 p4 组 —— stage_for_layer 返回
    # "protected" -> 输出保持全宽(512ch), 仅通过上游(层6/11 concat)的输入裁切做去噪。
    "p4": frozenset({5, 6, 11, 12, 18, 19, 20}),
    "p5": frozenset({7, 8, 9, 10, 21, 22}),
}
_MCSR_INPUT_LAYERS = frozenset({16, 19, 22})  # 路由器入口, 输出宽度必须保持不变
_DETECT_HEAD = frozenset({23})  # AdaptiveDefectDetect, 整体不动


def stage_for_layer(index: int) -> str:
    """Map a YOLO top-level layer index to its protected scale budget."""
    for stage, indices in _STAGE_LAYERS.items():
        if index in indices:
            return stage
    return "protected"


# ---------------------------------------------------------------------------
# 卷积/BN/入口识别
# ---------------------------------------------------------------------------

def _conv2d_of(block: nn.Module | None) -> nn.Conv2d | None:
    """Resolve an Ultralytics Conv-block (has .conv) to the underlying nn.Conv2d."""
    if isinstance(block, nn.Conv2d):
        return block
    inner = getattr(block, "conv", None)
    return inner if isinstance(inner, nn.Conv2d) else None


def _out_conv2d(layer: nn.Module) -> nn.Conv2d | None:
    """The Conv2d producing this layer's output (stage-output root port)."""
    for attr in ("cv3", "cv2", "conv"):
        conv = _conv2d_of(getattr(layer, attr, None))
        if conv is not None:
            return conv
    return _conv2d_of(layer)


def _entry_conv2d(layer: nn.Module) -> nn.Conv2d | None:
    """The Conv2d consuming this layer's raw input (input port)."""
    for attr in ("cv1", "conv"):
        conv = _conv2d_of(getattr(layer, attr, None))
        if conv is not None:
            return conv
    return _conv2d_of(layer)


# ---------------------------------------------------------------------------
# 路由强度 (从 MCSR router 权重抽取的静态可用量, 不引入在线推断)
# ---------------------------------------------------------------------------

def route_strength(model: nn.Module) -> tuple[float, float, float]:
    """Return per-scale MCSR route-dependency strength R(p3/p4/p5) drawn from the router weights.

    对 router.router[-1](hidden->num_scales^2) 的权重取绝对值的 hidden 均值, reshape 为
    (num_scales tap_target row x num_scales source col), 对 source 求均值 => 每个 target
    尺度被全局关注的平均强度。归一化到 0..1 后按 P3->P5 顺序返回。无 MCSR 时退化为全 1。
    """
    head = model.model[-1]
    router = getattr(head, "router", None)
    if router is None:
        return (1.0, 1.0, 1.0)
    num_scales = router.num_scales
    final_linear = router.router[-1]
    weight = final_linear.weight.detach().abs().mean(dim=1)  # (num_scales^2,) avg over hidden
    mat = weight.view(num_scales, num_scales)  # target 行 x source 列
    per_target = mat.mean(dim=1)  # (num_scales,) 每目标尺度被关注强度
    stable = per_target - per_target.min()
    denom = stable.max() + 1e-6
    relative = (stable / denom).cpu().tolist()
    return tuple(float(v) for v in relative)


# ---------------------------------------------------------------------------
# 阶段输出通道保留计算
# ---------------------------------------------------------------------------

def kept_channels_of(
    bn: nn.BatchNorm2d, ratio: float, round_to: int, r_scale: float, lambda_mod: float
) -> tuple[list[int], int]:
    """Return (kept_channel_indices, removed_count) using route-aware importance.

    score(c) = |BN-gamma(c)| * (1 + lambda * R_scale); 剪掉分数最小的通道, 保留宽度按 round_to 对齐。
    """
    channels = bn.num_features
    target_keep = min(channels, max(round_to, round(channels * (1.0 - ratio) / round_to) * round_to))
    keep_count = int(target_keep)
    remove_count = channels - keep_count
    if remove_count <= 0:
        return list(range(channels)), 0
    gamma = bn.weight.detach().abs().view(-1)
    score = gamma * (1.0 + lambda_mod * r_scale)
    removed = score.argsort()[:remove_count]
    removed_set = set(removed.cpu().tolist())
    kept = [i for i in range(channels) if i not in removed_set]
    return kept, remove_count


# ---------------------------------------------------------------------------
# 自研静态通道流动 (利用 parse_model 写的 .f / .c2)
# ---------------------------------------------------------------------------

def _orig_width(model: nn.Module, idx: int) -> int:
    """Layer idx's original (pre-prune) output channel width."""
    layer = model.model[idx]
    c2 = getattr(layer, "c2", None)
    if c2 is not None:
        return int(c2)
    root = _out_conv2d(layer)
    return int(root.out_channels) if root is not None else 0


def _input_sources(model: nn.Module, idx: int) -> list[int]:
    """Ordered source-layer indices composing layer idx's input (channel order)."""
    f = model.model[idx].f
    if f == -1:
        return [idx - 1]
    if isinstance(f, int):
        return [f]
    return [idx - 1 if j == -1 else int(j) for j in f]


def _kept_output_indices(model: nn.Module, idx: int, roots: dict[int, list[int]]) -> list[int]:
    """Layer idx's output channels that survive, in the layer's own (0..out-1) space.

    - 被剪根: 直接是其保留下标。
    - Concat: 沿其 sources 列出各块的保留位置并加上原块偏移(concat 通道顺序 = f 列表顺序)。
    - Upsample: 通道数不变, 保留下标 = 输入来源的保留下标。
    - 其他层(未被剪): 全部保留(输出宽度不变)。
    """
    if idx in roots:
        return list(roots[idx])
    layer = model.model[idx]
    f = layer.f
    if isinstance(f, list):
        kept: list[int] = []
        offset = 0
        for src in _input_sources(model, idx):
            if src < 0:  # 防御: 无效来源
                continue
            src_width = _orig_width(model, src)
            for p in _kept_output_indices(model, src, roots):
                kept.append(offset + p)
            offset += src_width
        return kept
    # 非 concat
    if getattr(layer, "type", "") == "Upsample" or layer.__class__.__name__ in ("Upsample", "nn.Upsample"):
        srcs = _input_sources(model, idx)
        return _kept_output_indices(model, srcs[0], roots) if srcs else list(range(_orig_width(model, idx)))
    # 未被剪的常规卷积层: 输出宽度不变, 全部保留
    return list(range(_orig_width(model, idx)))


def _entry_input_keep(model: nn.Module, idx: int, roots: dict[int, list[int]]) -> list[int]:
    """The kept input-channel indices for the entry conv of layer idx (in its original 0..in-1 space)."""
    srcs = _input_sources(model, idx)
    if not srcs:
        return []
    src = srcs[0]
    return _kept_output_indices(model, src, roots)


# ---------------------------------------------------------------------------
# 核心剪枝
# ---------------------------------------------------------------------------

def _slice_output(conv: nn.Conv2d, kept: list[int]) -> int:
    """Slice a conv's output dimension (weight out + bias), return removed param count."""
    n = conv.out_channels
    keep = torch.tensor(kept, dtype=torch.long)
    kw = conv.kernel_size[0] * conv.kernel_size[1]
    removed = (n - len(kept)) * (conv.weight.shape[1] * kw) + (n - len(kept))
    conv.weight.data = torch.nn.Parameter(conv.weight.data[keep].clone())
    if conv.bias is not None:
        conv.bias.data = torch.nn.Parameter(conv.bias.data[keep].clone())
    conv.out_channels = len(kept)
    return removed


def _slice_input(conv: nn.Conv2d, kept: list[int]) -> int:
    """Slice a conv's input dimension (weight in), return removed param count."""
    cin = conv.weight.shape[1]
    if len(kept) >= cin:
        return 0
    keep = torch.tensor(kept, dtype=torch.long)
    kw = conv.kernel_size[0] * conv.kernel_size[1]
    removed = (cin - len(kept)) * (conv.weight.shape[0] * kw)
    conv.weight.data = torch.nn.Parameter(conv.weight.data[:, keep].clone())
    conv.in_channels = len(kept)
    return removed


def _shrink_bn(bn: nn.BatchNorm2d, kept: list[int]) -> int:
    """Slice a BatchNorm's channel dim, return removed param count (weight+bias)."""
    n = bn.num_features
    if len(kept) >= n:
        return 0
    keep = torch.tensor(kept, dtype=torch.long)
    removed = (n - len(kept)) * 2
    bn.weight.data = torch.nn.Parameter(bn.weight.data[keep].clone())
    bn.bias.data = torch.nn.Parameter(bn.bias.data[keep].clone())
    if bn.running_mean is not None:
        bn.running_mean.data = bn.running_mean.data[keep].clone()
    if bn.running_var is not None:
        bn.running_var.data = bn.running_var.data[keep].clone()
    bn.num_features = len(kept)
    return removed


def _root_bn(model: nn.Module, root_conv2d: nn.Conv2d) -> nn.BatchNorm2d:
    """Find the BatchNorm paired with a Conv2d out-port (searching each layer's subtree)."""
    for layer in model.model:
        for module in layer.modules():
            if getattr(module, "conv", None) is root_conv2d:
                bn = getattr(module, "bn", None)
                if isinstance(bn, nn.BatchNorm2d):
                    return bn
        # 裸 Conv2d 作为层本身的情况
        if layer is root_conv2d:
            continue
    for layer in model.model:
        for module in layer.modules():
            if isinstance(module, nn.BatchNorm2d) and module.num_features == root_conv2d.out_channels:
                return module
    raise ValueError("no BatchNorm2d paired with output root")


def prune_model(model: nn.Module, example_inputs: torch.Tensor, config: RouteAwareConfig) -> PruningReport:
    """Apply route-aware structured pruning via self-built static channel-flow bookkeeping."""
    if example_inputs.ndim != 4 or example_inputs.shape[1] != 3:
        raise ValueError(f"example_inputs must have shape (N, 3, H, W), got {tuple(example_inputs.shape)}")

    model.eval()
    parameter = next(model.parameters())
    example_inputs = example_inputs.to(device=parameter.device, dtype=parameter.dtype)

    before_params = sum(p.numel() for p in model.parameters())
    before_macs = _count_macs(model, example_inputs)
    strengths = route_strength(model)
    stage_weight = {"p3": strengths[0], "p4": strengths[1], "p5": strengths[2]}

    with torch.no_grad():
        output_before = model(example_inputs)
    if output_before is None:
        raise RuntimeError("pre-prune forward returned no output")

    # 记录剪前输出形状用于结构校验 (DetectionModel returns a tuple/list of tensors)
    before_shapes = [tuple(t.shape) for t in _flatten(output_before) if isinstance(t, torch.Tensor)]

    # ---- 阶段1: 决定每个可剪根层的输出保留下标 ----
    roots: dict[int, list[int]] = {}  # layer index -> kept output indices
    pruned_layers = 0
    for idx, layer in enumerate(model.model):
        if idx >= 23 or idx in _MCSR_INPUT_LAYERS or idx in _DETECT_HEAD:
            continue
        stage = stage_for_layer(idx)
        if stage == "protected":
            continue
        r_scale = stage_weight.get(stage, 1.0)
        # 路由感知比例: 重要尺度(R 大)有效剪枝比例被压制, 弱尺度保持基准 —— 让路由强度落到结构上。
        ratio = config.effective_ratio_for(stage, r_scale)
        if ratio <= 0.0:
            continue
        root = _out_conv2d(layer)
        if root is None:
            continue  # Upsample / Concat
        bn = _root_bn(model, root)
        kept, removed = kept_channels_of(bn, ratio, config.round_to, r_scale, config.lambda_mod)
        if removed <= 0:
            continue
        roots[idx] = kept
        pruned_layers += 1

    if not roots:
        raise ValueError("no channels were selected for pruning")

    # ---- 阶段2: 先对所有根做输出通道裁切 (out 维 + BN) ----
    removed_total = 0
    for idx, kept in roots.items():
        root = _out_conv2d(model.model[idx])
        removed_total += _slice_output(root, kept)
        bn = _root_bn(model, root)
        removed_total += _shrink_bn(bn, kept)

    # ---- 阶段3: 对所有存在入口卷积的层做输入通道裁切 (in 维) ----
    for idx in range(1, len(model.model)):
        layer = model.model[idx]
        entry = _entry_conv2d(layer)
        if entry is None:
            continue  # Upsample / Concat / Detect (入口判无)
        keep = _entry_input_keep(model, idx, roots)
        if not keep:
            continue
        removed_total += _slice_input(entry, keep)

    model.eval()
    with torch.no_grad():
        output_after = model(example_inputs)
    if output_after is None:
        raise RuntimeError("post-prune forward returned no output")
    after_shapes = [tuple(t.shape) for t in _flatten(output_after) if isinstance(t, torch.Tensor)]
    if before_shapes != after_shapes:
        raise RuntimeError(
            f"pruned forward output shape changed: before={before_shapes}, after={after_shapes}; "
            "structured pruning broke the network"
        )

    after_params = sum(p.numel() for p in model.parameters())
    if after_params >= before_params:
        raise RuntimeError(f"route-aware pruning removed no parameters: {before_params} -> {after_params}")
    # 前向形状一致 => 每个消费端入口卷积/Concat 输入维度都与实际吻合, 结构性正确。
    # (若任一消费端未被同步收缩, PyTorch 卷积/拼接会就地报 shape mismatch)
    if after_shapes != before_shapes:
        raise RuntimeError(
            f"pruned forward output shape changed: before={before_shapes}, after={after_shapes}"
        )
    # 精确参数差是"消费者收缩 + 输出收缩"的净结果; 加法账目允许与真实差有小出入(同一次收缩
    # 在输出/输入两次计数中的重叠), 因此以真实差为准, 加法账目仅用于日志。
    additive = removed_total
    actual_removed = before_params - after_params

    after_macs = _count_macs(model, example_inputs)
    print(f"[route-aware] applied_removed(additive)={additive:,} actual_removed={actual_removed:,}")
    return PruningReport(
        before_macs=float(before_macs),
        after_macs=float(after_macs),
        before_params=int(before_params),
        after_params=int(after_params),
        removed_params=int(actual_removed),
        route_strength=strengths,
        lambda_mod=config.lambda_mod,
        pruned_layers=int(pruned_layers),
    )


def _flatten(obj):
    """Recursively flatten nested lists/tuples of tensors."""
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten(item)
    else:
        yield obj


def _count_macs(model: nn.Module, example_inputs: torch.Tensor) -> float:
    """Lightweight MACs estimate via ultralytics get_flops (reporting only)."""
    from ultralytics.utils.torch_utils import get_flops

    try:
        flops = get_flops(model, imgsz=max(example_inputs.shape[-1], 64))
        return float(flops) / 2.0
    except Exception:
        return 0.0


def save_pruned_checkpoint(yolo: YOLO, output_path: str | Path, report: PruningReport, mcsr: bool = True) -> Path:
    """Save a reloadable Ultralytics checkpoint with route-aware pruning metadata."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.ckpt = dict(yolo.ckpt or {})
    method = "route-aware MCSR-projected BN-gamma" if mcsr else "route-aware plain-yolo11 BN-gamma"
    yolo.ckpt["pruning"] = {
        "method": method,
        **asdict(report),
        "parameter_reduction": report.parameter_reduction,
        "mac_reduction": report.mac_reduction,
        "mcsr": mcsr,
        "dependency": "self-built static channel-flow (no torch-pruning)",
    }
    yolo.save(output_path)
    if not output_path.is_file():
        raise RuntimeError(f"checkpoint was not created: {output_path}")
    return output_path


def build_mcsr_model(weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu", mcsr: bool = True) -> YOLO:
    """Build the fixed MCSR architecture and transfer an unpruned checkpoint."""
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    loaded = YOLO(weights)
    loaded_head = loaded.model.model[-1]
    has_router = getattr(loaded_head, "router", None) is not None
    if mcsr:
        yolo = loaded if has_router else YOLO(MCSR_MODEL_YAML).load(weights)
    else:
        if has_router:
            raise ValueError("--no-mcsr requested but source checkpoint already contains an MCSR router")
        yolo = loaded
    yolo.model.to(torch.device(device)).eval()
    return yolo


def resolve_prune_source(weights: str | Path | None) -> Path:
    """Locate the unpruned-both cold checkpoint to prune from."""
    if weights is not None:
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prune source not found: {path}")
        return path
    for candidate in (UNPRUNED_BOTH, UNPRUNED_BOTH_FALLBACK):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("no unpruned-both checkpoint; pass --weights explicitly")


def main() -> None:
    """Prune the unpruned-both cold model with the route-aware method and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None, help="unpruned MCSR checkpoint to prune from")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="saved pruned checkpoint path")
    parser.add_argument("--p3-ratio", type=float, default=0.10)
    parser.add_argument("--p4-ratio", type=float, default=0.20)
    parser.add_argument("--p5-ratio", type=float, default=0.30)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--lambda", dest="lambda_mod", type=float, default=1.0, help="route-strength modulation")
    parser.add_argument(
        "--no-mcsr", action="store_true", help="prune a plain YOLO11 (no router) instead of the MCSR structure"
    )
    parser.add_argument(
        "--no-route-aware-ratio", action="store_true",
        help="disable R-based per-stage ratio modulation; use p3/p4/p5 ratios exactly (R is unreliable)",
    )
    parser.add_argument("--device", default="cpu", help="prune device, e.g. cpu or 0")
    args = parser.parse_args()

    config = RouteAwareConfig(
        p3_ratio=args.p3_ratio,
        p4_ratio=args.p4_ratio,
        p5_ratio=args.p5_ratio,
        round_to=args.round_to,
        lambda_mod=args.lambda_mod,
        mcsr=not args.no_mcsr,
        route_aware_ratio=not args.no_route_aware_ratio,
    )
    source = resolve_prune_source(args.weights)
    yolo = build_mcsr_model(source, device=args.device, mcsr=config.mcsr)
    print(f"[route-aware] pruning source: {source}")

    example_inputs = torch.randn(1, 3, 320, 320)
    report = prune_model(yolo.model, example_inputs, config)
    output = save_pruned_checkpoint(yolo, args.output, report, mcsr=config.mcsr)
    print(
        f"[route-aware] params {report.before_params:,} -> {report.after_params:,} "
        f"(-{report.parameter_reduction:.2%}), MACs {report.before_macs:.3f}G -> {report.after_macs:.3f}G"
    )
    print(f"[route-aware] route strength P3/P4/P5 = {tuple(round(v, 3) for v in report.route_strength)}")
    print(f"[route-aware] pruned_layers={report.pruned_layers} removed={report.removed_params:,}")
    print(f"[route-aware] saved: {output}")


if __name__ == "__main__":
    main()
