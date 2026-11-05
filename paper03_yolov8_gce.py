"""YOLOv8-GCE 论文单文件复现（Electronics 2024, 13, 173）— 不改动 ultralytics 仓库任何代码。

论文：YOLOv8-GCE: 面向锂电池极片缺陷检测的轻量化改进（GhostConv + GhostC2f + CA + EIoU）
  - GhostConv 替换 neck 中标准卷积；GhostC2f（C2f 的 bottleneck 全部换为 GhostBottleneck）替换 neck 中 C2f；
  - Coordinate Attention（CA）置于 neck 各尺度 GhostC2f 之后（论文消融表 5："behind the neck's GhostC2f" 最优）；
  - EIoU 损失替换 CIoU（公式 10：Loss = 1-IoU + rho^2(b,bgt)/c^2 + rho^2(w,wgt)/wc^2 + rho^2(h,hgt)/hc^2）。
  - 基座 YOLOv8n（backbone 保持 stock），数据集 = 当前电池表面缺陷数据集 detection_yolo_v2（3 类）。

单文件注入机制（不改源码）：
  1) 自定义模块类注册进 ultralytics.nn.tasks 命名空间（parse_model 用 globals()[m] 解析 YAML 模块名）；
  2) 重新赋值 tasks.base_modules / tasks.repeat_modules（模块级全局名，parse_model 每次调用时现查）使
     GhostC2f 获得与内置 C2f 完全相同的通道注入 + repeat 语义；
  3) monkey-patch ultralytics.utils.loss.BboxLoss.forward：CIoU=True -> EIoU=True
     （复用内置 metrics.bbox_iou 的 EIoU 实现，公式与论文一致）；
  4) 模型 YAML 由脚本内嵌字符串动态生成并写入临时文件，YOLO(yaml) 构建。

训练协议与当前目录 train.py 一致：
  75ep / imgsz640 / batch16 / workers4 / seed0 / deterministic / warmup_epochs1.0 /
  close_mosaic10（default.yaml）/ randaugment / erasing0.4 / cls_remap。

用法：
  python paper03_yolov8_gce.py                  # 正式训练 75ep
  python paper03_yolov8_gce.py --epochs 1       # 冒烟 1ep
  python paper03_yolov8_gce.py --name gce --exist-ok
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, GhostBottleneck, GhostConv
from ultralytics.utils.torch_utils import init_seeds

# ============================================================
# 1) 论文模块实现
# ============================================================

class GhostC2f(nn.Module):
    """GhostC2f：与 C2f 完全同构，仅把 Bottleneck 全部替换为 GhostBottleneck（论文 Figure 4）。

    通过 _patch_parse_model() 注册进 parse_model 的 base_modules/repeat_modules，
    因此获得与内置 C2f 完全相同的通道注入：GhostC2f(c1, c2, n, shortcut)。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道
        n (int): GhostBottleneck 数量
        shortcut (bool): 保留以兼容 C2f 调用（GhostBottleneck 自带恒等分支）
        g (int): 分组数（保留，未使用）
        e (float): 隐藏通道扩张比
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GhostC2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CoordAtt(nn.Module):
    """Coordinate Attention（Hou et al., CVPR 2021）：论文 03 用于 neck 各尺度 GhostC2f 之后的注意力模块。

    输入输出通道相等；通道数在首次 forward 时按输入自适应（reduction 压缩比）。
    不依赖 parse_model 的通道注入（非 base module），因此 YAML 中仅写 [-1, 1, CoordAtt, []]。
    """

    def __init__(self, reduction: int = 32):
        super().__init__()
        self.reduction = int(reduction)
        self._attn = None  # lazy-built (Conv2d -> BN -> SiLU -> Conv2d -> Sigmoid)

    def _get_attn(self, c: int, device: torch.device):
        if self._attn is None:
            r = max(c // self.reduction, 8)
            self._attn = nn.Sequential(
                nn.Conv2d(c, r, 1, bias=False),
                nn.BatchNorm2d(r),
                nn.SiLU(inplace=True),
                nn.Conv2d(r, c, 1, bias=False),
                nn.Sigmoid(),
            ).to(device)
        return self._attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = self._get_attn(c, x.device)
        x_h = x.mean(dim=3, keepdim=True)  # (b,c,h,1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (b,c,w,1)
        y = torch.cat([x_h, x_w], dim=2)  # (b,c,h+w,1)
        y = attn(y)  # attention weights in (0,1)
        a_h, a_w = y.split([h, w], dim=2)
        a_w = a_w.permute(0, 1, 3, 2)  # (b,c,1,w)
        return x * a_h * a_w


# ============================================================
# 2) 单文件注入：注册模块 + patch parse_model（base_modules/repeat_modules）
# ============================================================
tasks.GhostC2f = GhostC2f
tasks.CoordAtt = CoordAtt


def _patch_parse_model() -> None:
    """把 GhostC2f 注入 parse_model 的 base_modules / repeat_modules（二者是函数内局部 frozenset）。

    base_modules 决定自定义模块获得内置模块的通道注入（c1,c2 = ch[f], args[0] 并做 width 缩放），
    repeat_modules 决定 n 参数作为 repeat 数展开。parse_model 源码中这两个 frozenset 内各有一个
    'C2f,' 项；通过源码字符串替换 + exec 生成同语义的新 parse_model，并覆盖 tasks.parse_model。
    """
    import inspect

    src = inspect.getsource(tasks.parse_model)
    marker = "            C2f,\n"
    parts = src.split(marker)
    if len(parts) != 3:  # 恰好出现在 base_modules 与 repeat_modules 各一次
        raise RuntimeError(f"unexpected parse_model source layout (C2f, count={len(parts) - 1})")
    new_src = (
        parts[0] + marker + "            GhostC2f,\n"
        + parts[1] + marker + "            GhostC2f,\n"
        + parts[2]
    )
    exec(compile(new_src, "<patched_parse_model>", "exec"), tasks.__dict__)
    print("[gce] parse_model patched: GhostC2f registered as base/repeat module")


_patch_parse_model()

# ============================================================
# 3) EIoU 损失：monkey-patch BboxLoss.forward（CIoU -> EIoU）
# ============================================================
def _patch_eiou_loss() -> None:
    """把 BboxLoss.forward 的 bbox_iou(CIoU=True) 替换为 EIoU=True，其余逻辑逐字保留。"""
    from ultralytics.utils import loss as U
    from ultralytics.utils.loss import bbox2dist
    from ultralytics.utils.metrics import bbox_iou

    def forward_eiou(
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
        """BboxLoss.forward 的 EIoU 版本（唯一改动：CIoU=True -> EIoU=True）。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, EIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

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

    U.BboxLoss.forward = forward_eiou
    print("[gce] BboxLoss.forward patched: CIoU -> EIoU")


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
    print("[gce] v8DetectionLoss patched: empty-GT batch guard")


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
    print("[gce] DetectionTrainer.label_loss_items patched: NaN/Inf -> 0 (results.csv 无 NaN)")


# ============================================================
# 4) 模型 YAML（内嵌，基于 yolov8n，neck 全换 GhostC2f/GhostConv + 3x CA）
# ============================================================
GCE_YAML = """# YOLOv8-GCE (Electronics 2024) - 单文件复现；基于 yolov8.yaml，scale=n
nc: 3
scales:
  n: [0.33, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, GhostC2f, [512]] # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, GhostC2f, [256]] # 15 (P3/8-small)
  - [-1, 1, CoordAtt, []] # 16 P3 注意力（neck GhostC2f 之后）
  - [-1, 1, GhostConv, [256, 3, 2]] # 17
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, GhostC2f, [512]] # 19 (P4/16-medium)
  - [-1, 1, CoordAtt, []] # 20 P4 注意力
  - [-1, 1, GhostConv, [512, 3, 2]] # 21
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, GhostC2f, [1024]] # 23 (P5/32-large)
  - [-1, 1, CoordAtt, []] # 24 P5 注意力
  - [[16, 20, 24], 1, Detect, [nc]] # Detect(P3,P4,P5)
"""


def build_model(seed: int = 0, pretrained: str = "yolov8n.pt"):
    """固定随机状态后构建 YOLOv8-GCE 模型并加载预训练权重。"""
    init_seeds(seed, deterministic=True)
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="paper03_gce_", dir=str(Path.cwd()))
    import os
    os.close(fd)
    Path(tmp).write_text(GCE_YAML, encoding="utf-8")
    model = YOLO(tmp)
    Path(tmp).unlink(missing_ok=True)
    # 先 warmup forward 触发 CoordAtt/GhostC2f 的 lazy 构建（参数落位），再加载预训练：
    # 此时新模块（ckpt 中不存在）被 intersect_dicts 跳过、随机初始化，backbone 正常按名迁移。
    with torch.no_grad():
        model.model(torch.zeros(1, 3, 640, 640))
    model.load(pretrained)
    return model


# ============================================================
# 5) 训练入口（协议与 train.py 一致）
# ============================================================
DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
DEFAULT_EPOCHS = 75
DEFAULT_WARMUP_EPOCHS = 1.0
PRETRAINED = "yolov8n.pt"


def main() -> None:
    p = argparse.ArgumentParser(
        description="YOLOv8-GCE (Electronics 2024) 单文件复现：GhostC2f + CA + EIoU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--name", type=str, default="paper03_gce")
    p.add_argument("--pretrained", type=str, default=PRETRAINED)
    p.add_argument("--exist-ok", action="store_true")
    args = p.parse_args()

    _patch_eiou_loss()
    _patch_empty_gt_loss()
    _patch_val_loss_nan_guard()
    model = build_model(seed=args.seed, pretrained=args.pretrained)
    model.add_callback("on_val_batch_end", _val_nan_guard_callback())
    n_l, n_p, n_g, flops = model.model.info(verbose=True, imgsz=args.imgsz)
    print(f"[paper03] YOLOv8-GCE layers={n_l} params={n_p / 1e6:.3f}M GFLOPs={flops:.3f}")

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
