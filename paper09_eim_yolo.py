"""EIM-YOLO 论文单文件复现（Applied Sciences 2025, 15, 9380）— 不改动 ultralytics 仓库任何代码。

论文：EIM-YOLO 面向新能源汽车电池充电口密封盖表面缺陷检测（YOLOv11 基座）。
  - C3UltraPConv（C3k2 的瓶颈卷积替换为 FasterNet PConv 部分卷积，CSP 结构）替换 backbone 4 处 C3k2；
  - RFAConv（Receptive Field Attention Convolution：全局注意力图 ⊗ 多尺度局部特征，式 1）替换
    backbone 的 P3/P4/P5 下采样 Conv（保持 stride/通道语义）；
  - BiFPN-Concat（可学习权重加权特征融合）替换 neck 4 处 Concat；
  - Inner-SIoU 损失（内框 ratio=0.7 + SIoU 角度/距离/形状代价，式 2-18）替换 CIoU。
  - 基座 YOLO11n（yolo11.yaml, scale=n），数据集 = 当前电池表面缺陷数据集 detection_yolo_v2（3 类）。
  注：论文"EIM"缩写全文未展开，按模型名处理；FourD 四头检测头按 P2 分支实现并默认启用，
  需要三头消融时使用 --three-head。

单文件注入机制（不改源码）：
  1) 自定义模块注册进 ultralytics.nn.tasks 命名空间（parse_model 用 globals()[m] 解析 YAML 模块名）；
  2) 源码字符串替换 + exec 生成补丁版 parse_model：base_modules/repeat_modules 注入 C3UltraPConv
     （获得 C2f 同款通道注入 + repeat 语义）、base_modules 注入 RFAConv、Concat 特殊分支扩展
     BiFPNConcat（c2 = sum(ch[x] for x in f)）；
  3) monkey-patch ultralytics.utils.loss.bbox_iou -> inner_siou（BboxLoss.forward 的调用点绑定在
     loss 模块命名空间，故 patch 点 = loss 模块）；
  4) 模型 YAML 由脚本内嵌字符串动态生成并写入临时文件，YOLO(yaml) 构建。

训练协议与当前目录 train.py 一致：
  75ep / imgsz640 / batch16 / workers4 / seed0 / deterministic / warmup_epochs1.0 /
  close_mosaic10（default.yaml）/ randaugment / erasing0.4 / cls_remap。

用法：
  python paper09_eim_yolo.py                  # 正式训练 75ep（论文完整 4 头）
  python paper09_eim_yolo.py --three-head     # 关闭 P2 分支，运行 3 头消融
  python paper09_eim_yolo.py --epochs 1       # 冒烟 1ep
  python paper09_eim_yolo.py --name eim --exist-ok
"""

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.utils.torch_utils import init_seeds

# ============================================================
# 1) 论文模块实现
# ============================================================

class PConv(nn.Module):
    """FasterNet Partial Convolution：仅对前 ratio 比例通道做卷积，其余通道恒等直通（c1==c2）。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道（== c1）
        k (int): 卷积核大小
        s (int): 步长
        ratio (float): 参与卷积的通道比例
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, ratio: float = 0.25, act: bool = True):
        super().__init__()
        assert c1 == c2, "PConv identity branch requires c1 == c2"
        self.c_ = max(1, int(c1 * ratio))
        self.conv = Conv(self.c_, self.c_, k, s, k // 2, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对部分通道卷积，其余恒等，concat 输出。"""
        x1, x2 = x[:, : self.c_], x[:, self.c_:]
        return torch.cat([self.conv(x1), x2], dim=1)


class C3UltraPConv(nn.Module):
    """C3UltraPConv：CSP 风格结构（论文 Sec. 2.2：标准卷积 -> 两支 PConv -> Concat -> 融合卷积）。

    以 C2f 布线为底座（cv1 降维 -> n 个 PConv 瓶颈 -> Concat -> cv2 融合），瓶颈替换为
    PConv(3x3) + Conv(1x1)（PConv 只算部分通道，显著降 FLOPs）。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道
        n (int): PConv 瓶颈数量
        c3k (bool): 兼容 C3k2 签名（未使用，保留）
        e (float): 隐藏通道扩张比
        shortcut (bool): 保留以兼容
        g (int): 分组数
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, shortcut: bool = True, g: int = 1):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(PConv(self.c, self.c, 3, 1, ratio=0.25), Conv(self.c, self.c, 1, act=True))
            for _ in range(max(n, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3UltraPConv layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RFAConv(nn.Module):
    """RFAConv（Receptive Field Attention Convolution，arXiv:2304.03198）：论文 09 Sec. 2.3。

    F = Softmax(g_1x1(AvgPool(X))) * ReLU(Norm(g_kxk(X)))
    全局注意力图 A_rf（GAP -> 1x1 conv -> softmax）逐元素乘以多尺度局部特征 F_rf
    （3/5/7 深度卷积并联求和 -> ReLU）。支持 stride 下采样与通道变换（可替换任意 Conv）。

    Args:
        c1 (int): 输入通道
        c2 (int): 输出通道
        k (int): 主卷积核（保留）
        s (int): 步长
        p (int): 填充
        ks (tuple): 多尺度局部核
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, ks: tuple = (3, 5, 7), act: bool = True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        # 步长下采样（stride>1 时先下采样，保证多尺度核输出同分辨率）
        self.down = Conv(c1, c1, k, s, p, g=c1, act=False) if s > 1 else nn.Identity()
        # 多尺度局部特征：深度卷积并联求和（同分辨率）
        self.local = nn.ModuleList(Conv(c1, c1, kk, 1, kk // 2, g=c1, act=False) for kk in ks)
        # 全局池化后空间尺寸固定为 1x1；这里不能使用带 BatchNorm 的 Conv，
        # 否则最后一个 batch 只有一张图时训练会因 BN 无法估计方差而中断。
        self.g1 = nn.Conv2d(c1, c1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.proj = Conv(c1, c2, 1, act=act) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RFAConv forward: A_rf * F_rf，然后（可选）1x1 投影到 c2。"""
        A_rf = torch.softmax(self.g1(x.mean(dim=(2, 3), keepdim=True)), dim=1)  # (B,C,1,1)
        xd = self.down(x)
        F_rf = self.act(sum(l(xd) for l in self.local))  # (B,C,H,W)
        return self.proj(A_rf * F_rf)


class BiFPNConcat(nn.Module):
    """BiFPN 可学习权重特征融合（论文 09 Sec. 2.4）：relu(w_i) 归一化后逐分支加权。

    mode='concat'：加权后拼接，输出通道 = 各分支之和（YAML 通道不变，最稳妥）；
    mode='sum'：加权求和（正统 BiFPN），输出通道 = 单分支通道。
    """

    def __init__(self, dimension: int = 1, mode: str = "concat"):
        super().__init__()
        self.d = dimension
        self.mode = mode
        self.w = nn.Parameter(torch.ones(4))
        self.eps = 1e-4

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """x: 待融合的特征张量列表（同分辨率）。"""
        w = torch.relu(self.w[: len(x)])
        w = w / (w.sum() + self.eps)
        if self.mode == "sum":
            return sum(wi * xi for wi, xi in zip(w, x))
        return torch.cat([wi * xi for wi, xi in zip(w, x)], dim=self.d)


# ============================================================
# 2) 单文件注入：注册模块 + patch parse_model
# ============================================================
tasks.C3UltraPConv = C3UltraPConv
tasks.RFAConv = RFAConv
tasks.BiFPNConcat = BiFPNConcat
# 允许自定义类被 torch.serialization safe_globals 白名单加载（PyTorch>=2.6 checkpoint resume）
try:
    torch.serialization.add_safe_globals([C3UltraPConv, PConv, RFAConv, BiFPNConcat])
except Exception:
    pass


def _patch_parse_model() -> None:
    """补丁版 parse_model：
    - base_modules：C3k2 行后注入 C3UltraPConv + RFAConv（获得 C2f 同款通道注入）；
    - repeat_modules：C3k2 行后注入 C3UltraPConv（repeat 展开语义）；
    - Concat 特殊分支扩展为 BiFPNConcat（c2 = sum(ch[x] for x in f)）。
    """
    import inspect

    src = inspect.getsource(tasks.parse_model)
    # 1) base_modules：第一个 '            C3k2,\n' 后插入
    marker = "            C3k2,\n"
    parts = src.split(marker)
    if len(parts) != 3:
        raise RuntimeError(f"unexpected parse_model source layout (C3k2, count={len(parts) - 1})")
    src = (
        parts[0] + marker + "            C3UltraPConv,\n            RFAConv,\n"
        + parts[1] + marker + "            C3UltraPConv,\n"
        + parts[2]
    )
    # 2) Concat 特殊分支
    src = src.replace("        elif m is Concat:", "        elif m is Concat or m is BiFPNConcat:")
    exec(compile(src, "<patched_parse_model>", "exec"), tasks.__dict__)
    print("[eim] parse_model patched: C3UltraPConv/RFAConv as base modules, BiFPNConcat in Concat branch")


_patch_parse_model()


# ============================================================
# 3) Inner-SIoU 损失：monkey-patch ultralytics.utils.loss.bbox_iou
# ============================================================
def inner_siou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    ratio: float = 0.7,
    theta: float = 4.0,
    eps: float = 1e-7,
    **kwargs,
) -> torch.Tensor:
    """Inner-SIoU（论文 09 式 2-18）：内框 IoU（ratio=0.7）+ SIoU 距离/角度/形状代价。

    化简后：L_Inner-SIoU = IoU_inner - 0.5*(distance_cost + shape_cost)（外层 IoU 抵消）。
    box1/box2: (..., 4) 以 (x1, y1, x2, y2) 或 xywh 格式输入。
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1x1, b1y1, b1x2, b1y2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        b2x1, b2y1, b2x2, b2y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
        w1, h1, w2, h2 = b1x2 - b1x1, b1y2 - b1y1, b2x2 - b2x1, b2y2 - b2y1
    else:
        b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
        b2x1, b2y1, b2x2, b2y2 = box2.chunk(4, -1)
        w1, h1 = b1x2 - b1x1, b1y2 - b1y1
        w2, h2 = b2x2 - b2x1, b2y2 - b2y1

    # --- Inner boxes（式 2-5）---
    cx1, cy1 = (b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2
    cx2, cy2 = (b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2
    iw1, ih1, iw2, ih2 = w1 * ratio, h1 * ratio, w2 * ratio, h2 * ratio
    ib1x1, ib1y1, ib1x2, ib1y2 = cx1 - iw1 / 2, cy1 - ih1 / 2, cx1 + iw1 / 2, cy1 + ih1 / 2
    ib2x1, ib2y1, ib2x2, ib2y2 = cx2 - iw2 / 2, cy2 - ih2 / 2, cx2 + iw2 / 2, cy2 + ih2 / 2
    inter = (torch.min(ib1x2, ib2x2) - torch.max(ib1x1, ib2x1)).clamp_(0) * (
        torch.min(ib1y2, ib2y2) - torch.max(ib1y1, ib2y1)
    ).clamp_(0)
    union = iw1 * ih1 + iw2 * ih2 - inter + eps
    iou_inner = inter / union  # 式 8

    # --- SIoU 代价（式 9-16）---
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    chh = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    scw, sch = torch.abs(cx2 - cx1), torch.abs(cy2 - cy1)
    sigma = torch.sqrt(scw.square() + sch.square())
    sin_alpha_1 = sch / (sigma + eps)
    sin_alpha_2 = scw / (sigma + eps)
    sin_alpha = torch.where(sin_alpha_1 > math.sqrt(0.5), sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha.clamp(0.0, 1.0)) * 2 - math.pi / 2)
    gamma = angle_cost - 2
    rho_x = (scw / (cw + eps)).square()
    rho_y = (sch / (chh + eps)).square()
    dist_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)  # 式 12-15
    omega_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + eps)
    omega_h = torch.abs(h1 - h2) / (torch.max(h1, h2) + eps)
    shape_cost = (1 - torch.exp(-omega_w)).pow(theta) + (1 - torch.exp(-omega_h)).pow(theta)  # 式 16

    # 式 17+18 化简：L = IoU_inner - 0.5*(dist + shape)
    return iou_inner - 0.5 * (dist_cost + shape_cost)


def _patch_inner_siou() -> None:
    """把 ultralytics.utils.loss 模块命名空间里的 bbox_iou 绑定替换为 inner_siou。

    BboxLoss.forward 通过 loss 模块全局名 bbox_iou 调用（loss.py line 18 的 import 绑定在 loss 命名空间），
    因此只 patch loss.bbox_iou 即可让训练损失生效。
    """
    from ultralytics.utils import loss as L

    L.bbox_iou = inner_siou
    print("[eim] loss.bbox_iou patched -> Inner-SIoU (ratio=0.7, theta=4)")


# ============================================================
# 4) 模型 YAML（内嵌，基于 yolo11.yaml scale=n）
# ============================================================
EIM_YAML_TMPL = """# EIM-YOLO (Applied Sciences 2025) 单文件复现；基于 yolo11.yaml，scale=n
nc: 3
scales:
  n: [0.50, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3UltraPConv, [256, False, 0.25]] # 2 (was C3k2)
  - [-1, 1, RFAConv, [256, 3, 2]] # 3-P3/8 (was Conv)
  - [-1, 2, C3UltraPConv, [512, False, 0.25]] # 4 (was C3k2)
  - [-1, 1, RFAConv, [512, 3, 2]] # 5-P4/16 (was Conv)
  - [-1, 2, C3UltraPConv, [512, True]] # 6 (was C3k2)
  - [-1, 1, RFAConv, [1024, 3, 2]] # 7-P5/32 (was Conv)
  - [-1, 2, C3UltraPConv, [1024, True]] # 8 (was C3k2)
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, BiFPNConcat, [1]] # 12 (was Concat)
  - [-1, 2, C3k2, [512, False]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, BiFPNConcat, [1]] # 15 (was Concat)
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, BiFPNConcat, [1]] # 18 (was Concat)
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, BiFPNConcat, [1]] # 21 (was Concat)
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)
{P2_LAYERS}
  - [[{DETECT_INPUTS}], 1, Detect, [nc]]
"""

P2_LAYERS = """  - [16, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 1], 1, Concat, [1]] # cat backbone P2/4
  - [-1, 2, C3k2, [128, False]] # P2/4 小目标头特征
"""
FOURD_INPUTS = "25, 16, 19, 22"  # P2, P3, P4, P5
THREE_INPUTS = "16, 19, 22"  # P3, P4, P5


def build_model(seed: int = 0, pretrained: str = "yolo11n.pt", fourd: bool = True):
    """固定随机状态后构建 EIM-YOLO 模型并加载预训练权重。"""
    init_seeds(seed, deterministic=True)
    yaml_text = EIM_YAML_TMPL.format(
        P2_LAYERS=P2_LAYERS if fourd else "",
        DETECT_INPUTS=FOURD_INPUTS if fourd else THREE_INPUTS,
    )
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="paper09_eim_", dir=str(Path.cwd()))
    import os
    os.close(fd)
    Path(tmp).write_text(yaml_text, encoding="utf-8")
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
TRAIN_RESULTS_DIR = r"C:/Users/23223/Documents/EIM_training_runs"
RESULTS_DIR = r"C:/Users/23223/Desktop/新论文数据保存/对比实验数据"


def copy_training_result(project: str, name: str) -> Path:
    """Copy a completed ASCII-path run to the requested comparison-results directory."""
    source = Path(project) / name
    destination = Path(RESULTS_DIR) / name
    if not source.is_dir():
        raise FileNotFoundError(f"训练结果目录不存在，无法复制：{source}")
    if destination.exists():
        raise FileExistsError(f"目标结果目录已存在，为避免覆盖不会复制：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"[paper09] 训练结果已复制到 {destination}")
    return destination


def main() -> None:
    p = argparse.ArgumentParser(
        description="EIM-YOLO (Applied Sciences 2025) 单文件复现：C3UltraPConv + RFAConv + BiFPN-Concat + Inner-SIoU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--name", type=str, default="paper09_eim_fixed_75ep")
    p.add_argument("--project", type=str, default=TRAIN_RESULTS_DIR, help="ASCII 临时训练目录")
    p.add_argument("--pretrained", type=str, default=PRETRAINED)
    p.add_argument("--three-head", dest="fourd", action="store_false", default=True, help="关闭 P2 分支，运行 3 头消融")
    p.add_argument("--exist-ok", action="store_true")
    args = p.parse_args()

    _patch_inner_siou()
    model = build_model(seed=args.seed, pretrained=args.pretrained, fourd=args.fourd)
    n_l, n_p, n_g, flops = model.model.info(verbose=True, imgsz=args.imgsz)
    print(f"[paper09] EIM-YOLO fourd={args.fourd} layers={n_l} params={n_p / 1e6:.3f}M GFLOPs={flops:.3f}")

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
        project=args.project,
        name=args.name,
    )
    copy_training_result(args.project, args.name)


if __name__ == "__main__":
    main()
