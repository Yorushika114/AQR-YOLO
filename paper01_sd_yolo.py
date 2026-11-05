"""SD-YOLO 论文单文件复现（Scientific Reports 2026, 16:1074）— 不改动 ultralytics 仓库任何代码。

论文：SD-YOLO 面向新能源汽车锂电池表面缺陷检测（YOLOv11n 基座）。
  - SimAM（无参数注意力，能量函数式）集成进 backbone：P3/P4/P5 的 C3k2 之后各加一处；
  - C3k2 GhostDynamicConv（CSP 结构 + C3k 块 + Ghost 模块 + DynamicConv 多核注意力加权）替换
    neck 中 4 处 C3k2（论文 Sec. "In the neck of the network, the C3k2 GhostDynamicConv..."）；
  - Focaler-IoU 混合损失（式 2-8：d=0.1/u=0.9 线性区间映射 + CIoU 几何项）替换纯 CIoU：
      IoU_focaler = clamp((IoU-d)/(u-d), 0, 1)；L_reg = 1 - IoU_focaler + (IoU - CIoU)。
  - 基座 YOLO11n（yolo11.yaml, scale=n），数据集 = 当前电池表面缺陷数据集 detection_yolo_v2（3 类）。
  注：论文 GhostDynamicConv 内部拓扑为位图无文字描述，本文按命名/引文（C3k + Ghost + DynamicConv）
  合理重建（★推测），结果与论文会有合理偏差。

单文件注入机制（不改源码）：
  1) 自定义模块注册进 ultralytics.nn.tasks 命名空间（parse_model 用 globals()[m] 解析 YAML 模块名）；
  2) 源码字符串替换 + exec 生成补丁版 parse_model：base_modules/repeat_modules 注入
     C3k2GhostDynamicConv（获得 C3k2 同款通道注入 + repeat 语义）；
  3) monkey-patch ultralytics.utils.loss.BboxLoss.forward：CIoU -> Focaler-IoU 混合损失
     （bbox_iou 的引用绑定在 loss 模块命名空间，patch loss 模块即可）；
  4) 模型 YAML 由脚本内嵌字符串动态生成并写入临时文件，YOLO(yaml) 构建。

训练协议与当前目录 train.py 一致：
  75ep / imgsz640 / batch16 / workers4 / seed0 / deterministic / warmup_epochs1.0 /
  close_mosaic10（default.yaml）/ randaugment / erasing0.4 / cls_remap。

用法：
  python paper01_sd_yolo.py                  # 正式训练 75ep
  python paper01_sd_yolo.py --epochs 1       # 冒烟 1ep
  python paper01_sd_yolo.py --name sd_yolo --exist-ok
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, Conv, GhostConv
from ultralytics.utils.torch_utils import init_seeds

# ============================================================
# 1) 论文模块实现
# ============================================================

class SimAM(nn.Module):
    """SimAM：无参数注意力（ICML 2021, arXiv:2111.07679）。基于神经元能量函数重标定特征。

    零参数开销：X_att = X * sigmoid(1/E)，E 由能量函数解析解给出。
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SimAM forward：对输入特征做能量注意力重标定（通道/空间不变）。"""
        b, c, h, w = x.shape
        n = h * w - 1
        x_minus_mu_sq = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class DynamicConv(nn.Module):
    """DynamicConv：多卷积核 + SE 式注意力权重加权求和（CondConv/ODConv 思想，论文 01 ★推测）。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道
        k (int): 卷积核大小
        s (int): 步长
        p (int): 填充
        num_kernels (int): 并行卷积核数
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, num_kernels: int = 4, act: bool = True):
        super().__init__()
        self.num_kernels = num_kernels
        self.convs = nn.ModuleList(
            nn.Conv2d(c1, c2, k, s, p, bias=False) for _ in range(num_kernels)
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        r = max(c1 // 16, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, num_kernels, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按输入动态加权聚合各卷积核输出。"""
        w = self.attn(x)  # (B, K, 1, 1)
        y = sum(wi.unsqueeze(-1) * conv(x) for wi, conv in zip(w.unbind(1), self.convs))
        return self.act(self.bn(y))


class C3kGhostDynamicConv(nn.Module):
    """C3k GhostDynamicConv（论文 01 Fig. 4，★推测）：C3k 风格瓶颈 + Ghost + DynamicConv 组合。

    结构：DynamicConv(1x1) 动态通道变换 -> GhostConv(3x3) 廉价特征生成（含恒等残差）。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道
        k (int): 卷积核大小（保留）
        shortcut (bool): 是否使用恒等残差
        g (int): 分组数（保留）
        e (float): 隐藏通道扩张比
    """

    def __init__(self, c1: int, c2: int, k: int = 3, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DynamicConv(c1, c_, 1, 1, 0)  # dynamic 1x1
        self.cv2 = GhostConv(c_, c2, 3, 1, 1)  # ghost 3x3
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """瓶颈前向：y = cv2(cv1(x)) + x（若 shortcut）。"""
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3k2GhostDynamicConv(nn.Module):
    """C3k2 GhostDynamicConv（论文 01 Fig. 3）：C3k2（C2f）布线 + C3kGhostDynamicConv 瓶颈。

    签名与 C3k2 对齐（c1, c2, n, c3k, e, g, shortcut），以便注入 parse_model 的
    base_modules/repeat_modules 后获得与 C3k2 完全相同的通道注入与 repeat 语义。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道
        n (int): 瓶颈数量
        c3k (bool): 兼容 C3k2 签名（未使用，瓶颈固定为 GhostDynamic 变体）
        e (float): 隐藏通道扩张比
        g (int): 分组数（保留）
        shortcut (bool): 是否使用恒等残差
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3kGhostDynamicConv(self.c, self.c, 3, shortcut, g, e=1.0) for _ in range(max(n, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3k2GhostDynamicConv layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================
# 2) 单文件注入：注册模块 + patch parse_model
# ============================================================
tasks.SimAM = SimAM
tasks.C3k2GhostDynamicConv = C3k2GhostDynamicConv
try:
    torch.serialization.add_safe_globals([SimAM, DynamicConv, C3kGhostDynamicConv, C3k2GhostDynamicConv])
except Exception:
    pass


def _patch_parse_model() -> None:
    """补丁版 parse_model：base_modules / repeat_modules 的 C3k2 行后注入 C3k2GhostDynamicConv。"""
    import inspect

    src = inspect.getsource(tasks.parse_model)
    marker = "            C3k2,\n"
    parts = src.split(marker)
    if len(parts) != 3:
        raise RuntimeError(f"unexpected parse_model source layout (C3k2, count={len(parts) - 1})")
    new_src = (
        parts[0] + marker + "            C3k2GhostDynamicConv,\n"
        + parts[1] + marker + "            C3k2GhostDynamicConv,\n"
        + parts[2]
    )
    exec(compile(new_src, "<patched_parse_model>", "exec"), tasks.__dict__)
    print("[sd-yolo] parse_model patched: C3k2GhostDynamicConv registered as base/repeat module")


_patch_parse_model()


# ============================================================
# 3) Focaler-IoU 损失：monkey-patch BboxLoss.forward（式 8）
# ============================================================
def focaler_map(iou: torch.Tensor, d: float = 0.1, u: float = 0.9) -> torch.Tensor:
    """Focaler-IoU 线性区间映射（式 2）：[0, d) -> 0；[d, u] -> (IoU-d)/(u-d)；(u, 1] -> 1。"""
    return ((iou - d) / (u - d)).clamp(0.0, 1.0)


def _patch_focaler_iou() -> None:
    """把 BboxLoss.forward 的 bbox_iou(CIoU=True) 替换为 Focaler-IoU 混合损失（式 8）。

    L_reg = 1 - IoU_focaler + (IoU - CIoU)，其中 CIoU 提供几何项（式 5：1-IoU + d²/c² + βv）。
    """
    from ultralytics.utils import loss as U
    from ultralytics.utils.loss import bbox2dist
    from ultralytics.utils.metrics import bbox_iou

    def forward_focaler(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BboxLoss.forward 的 Focaler-IoU 版本（唯一改动：IoU 损失项）。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou_raw = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)  # IoU
        iou_ciou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)  # CIoU
        iou_focaler = focaler_map(iou_raw, d=0.1, u=0.9)
        loss_iou = ((1.0 - iou_focaler + (iou_raw - iou_ciou)) * weight).sum() / target_scores_sum  # 式 8

        # DFL loss（与 stock 完全一致）
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                torch.nn.functional.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    U.BboxLoss.forward = forward_focaler
    print("[sd-yolo] BboxLoss.forward patched: CIoU -> Focaler-IoU hybrid (d=0.1, u=0.9)")


def _patch_empty_gt_loss() -> None:
    """空 GT batch 保护：v8DetectionLoss 对无目标 batch（val 的 background 批次）维度崩溃，
    导致 val/box_loss 等记录为 NaN。包装 get_assigned_targets_and_loss，空 GT 时返回零 loss。"""
    from ultralytics.utils import loss as U

    _orig = U.v8DetectionLoss.get_assigned_targets_and_loss

    def _safe(self, preds, batch):
        n_inst = batch["cls"].numel()
        if n_inst == 0:
            loss = torch.zeros(3, device=self.device)
            return (None, loss, loss.detach())
        return _orig(self, preds, batch)

    U.v8DetectionLoss.get_assigned_targets_and_loss = _safe
    print("[sd-yolo] v8DetectionLoss patched: empty-GT batch guard")


def _val_nan_guard_callback():
    """on_val_batch_end 回调：val loss 若含 NaN/Inf 置 0（防 results.csv 记录 NaN 影响画图）。"""
    def _guard(trainer):
        v = getattr(trainer, "validator", None)
        if v is not None and torch.is_tensor(getattr(v, "loss", None)):
            if not torch.isfinite(v.loss).all():
                v.loss = torch.nan_to_num(v.loss)

    return _guard


def _patch_val_loss_nan_guard() -> None:
    """把 DetectionTrainer.label_loss_items 包装：loss_items 含 NaN/Inf 时先置 0（results.csv 无 NaN）。"""
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    _orig = DetectionTrainer.label_loss_items

    def _safe(self, loss_items=None, prefix="train"):
        if loss_items is not None:
            if torch.is_tensor(loss_items):
                loss_items = torch.nan_to_num(loss_items)
            else:
                loss_items = [0.0 if (x != x) else float(x) for x in loss_items]
        return _orig(self, loss_items, prefix)

    DetectionTrainer.label_loss_items = _safe
    print("[sd-yolo] DetectionTrainer.label_loss_items patched: NaN/Inf -> 0 (results.csv 无 NaN)")


# ============================================================
# 4) 模型 YAML（内嵌，基于 yolo11.yaml scale=n + SimAM + neck GhostDynamicConv）
# ============================================================
SD_YAML = """# SD-YOLO (Scientific Reports 2026) 单文件复现；基于 yolo11.yaml，scale=n
nc: 3
scales:
  n: [0.50, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]] # 4
  - [-1, 1, SimAM, []] # 5 P3 注意力（backbone 集成）
  - [-1, 1, Conv, [512, 3, 2]] # 6-P4/16
  - [-1, 2, C3k2, [512, True]] # 7
  - [-1, 1, SimAM, []] # 8 P4 注意力
  - [-1, 1, Conv, [1024, 3, 2]] # 9-P5/32
  - [-1, 2, C3k2, [1024, True]] # 10
  - [-1, 1, SimAM, []] # 11 P5 注意力
  - [-1, 1, SPPF, [1024, 5]] # 12
  - [-1, 2, C2PSA, [1024]] # 13
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 6], 1, Concat, [1]] # 15 cat backbone P4
  - [-1, 2, C3k2GhostDynamicConv, [512, False]] # 16
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 17
  - [[-1, 4], 1, Concat, [1]] # 18 cat backbone P3
  - [-1, 2, C3k2GhostDynamicConv, [256, False]] # 19 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]] # 20
  - [[-1, 16], 1, Concat, [1]] # 21 cat head P4
  - [-1, 2, C3k2GhostDynamicConv, [512, False]] # 22 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]] # 23
  - [[-1, 13], 1, Concat, [1]] # 24 cat head P5
  - [-1, 2, C3k2GhostDynamicConv, [1024, True]] # 25 (P5/32-large)
  - [[19, 22, 25], 1, Detect, [nc]] # 26 Detect(P3,P4,P5)
"""


def build_model(seed: int = 0, pretrained: str = "yolo11n.pt"):
    """固定随机状态后构建 SD-YOLO 模型并加载预训练权重。"""
    init_seeds(seed, deterministic=True)
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="paper01_sd_", dir=str(Path.cwd()))
    import os
    os.close(fd)
    Path(tmp).write_text(SD_YAML, encoding="utf-8")
    model = YOLO(tmp)
    Path(tmp).unlink(missing_ok=True)
    model.load(pretrained)
    return model


# ============================================================
# 5) 训练入口（协议与 train.py 一致）
# ============================================================
DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
DEFAULT_EPOCHS = 75
DEFAULT_WARMUP_EPOCHS = 1.0
PRETRAINED = "yolo11n.pt"


def main() -> None:
    p = argparse.ArgumentParser(
        description="SD-YOLO (Scientific Reports 2026) 单文件复现：SimAM + C3k2-GhostDynamicConv + Focaler-IoU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--name", type=str, default="paper01_sd_yolo")
    p.add_argument("--pretrained", type=str, default=PRETRAINED)
    p.add_argument("--exist-ok", action="store_true")
    args = p.parse_args()

    _patch_focaler_iou()
    _patch_empty_gt_loss()
    _patch_val_loss_nan_guard()
    model = build_model(seed=args.seed, pretrained=args.pretrained)
    model.add_callback("on_val_batch_end", _val_nan_guard_callback())
    n_l, n_p, n_g, flops = model.model.info(verbose=True, imgsz=args.imgsz)
    print(f"[paper01] SD-YOLO layers={n_l} params={n_p / 1e6:.3f}M GFLOPs={flops:.3f}")

    model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        warmup_epochs=DEFAULT_WARMUP_EPOCHS,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        save=True,
        resume=False,
        val=True,
        exist_ok=args.exist_ok,
        name=args.name,
    )


if __name__ == "__main__":
    main()
