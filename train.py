"""本地训练入口：YOLO11n 电池表面缺陷检测，唯一保留创新点（MCSR 结构路由 + SA-TAL 尺寸感知分配）的统一开关。

用法（命令行）:
    python train.py                     # 默认：文件内开关 RUN_ABLATION（当前 = "both"）
    python train.py --ablation mcsr     # 只开 MCSR（yolo11n-adaptive-defect.yaml），sa_tal=False
    python train.py --ablation sa_tal   # 只开 SA-TAL（base yolo11.yaml），sa_tal=True
    python train.py --ablation both     # MCSR + SA-TAL 组合（达标主臂）
    python train.py --epochs 50 --name my_run  # 普通模式（默认全关，纯 stock）

说明:
    - 基线 = 远程纯 stock（sa_ctl，Gemini 服务器）：mAP50-95 = 0.72445
      （50轮/640/batch16/seed0/deterministic，warmup_epochs=1.0，close_mosaic=10，
      auto_augment=randaugment，erasing=0.4，cls_remap=true）。基线不本地复跑，直接以此数为参照。
    - 已去除全部其他创新模块：DASA、QFL(iou_aware)、P2、LCC(crack_w)、crack_floor、DFL 等比/等频/
      逐层布局、NWD、α-IoU、scol、EIoU、consensus_rerank、CA/C3k2CA/C3k2SimAM/C3k2Deform、AR-Prior、
      TAL beta/topk 透传、蒸馏接入。历史结论见桌面《创新点研究与失败沉淀》。
    - MCSR（MorphologyConditionedScaleRouter）= 结构级开关：用 yolo11n-adaptive-defect.yaml 即激活
      （AdaptiveDefectDetect 头内，对 P3/P4/P5 形态学描述子路由跨尺度残差消息）。恒等初始化
      （residual_gain=0）→ 初始零扰动、预训练全量迁移。mcsr=False 用 base yaml（纯 stock Detect）。
    - SA-TAL（Size-Aware TaskAlignedAssigner）= loss 侧 hyp 开关：sa_tal=True 时对小 GT 框（对角线
      < sa_size px）拓宽候选椭圆并混合尺度不变的 center-quality 项。sa_tal=False 时 byte-identical stock TAL。
    - 两臂正交：mcsr=yaml 开关，sa_tal=loss hyp 开关。达标目标：基线 +0.05~0.10（mAP50-95 0.774~0.824）。
    - 结果保存于 runs/detect/<name>/。
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds

# 静默 TQDM 进度条（trainer 内部每 batch 刷屏），只保留 LOGGER 输出。
# 效果：训练日志里每轮只留下验证表格（LOGGER.info）与进度摘要，不再被 batch 进度条刷屏。
def quiet_tqdm() -> None:
    """把 ultralytics 的 TQDM 进度条强制 disable，验证 per-class 表格（走 LOGGER）不受影响。"""
    from ultralytics.utils import tqdm as _utqdm

    _orig_init = _utqdm.TQDM.__init__

    def _quiet_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _orig_init(self, *args, **kwargs)

    _utqdm.TQDM.__init__ = _quiet_init


quiet_tqdm()

EXPERIMENT_SEED = 0
# 轮数的单一事实来源。修改训练轮数只需改这里（或命令行传 --epochs N，后者优先）。
# 用户指令（2026-08-06）：拉高消融轮数至 75 轮裁决（基线仍为远程 50 轮记录 0.72445）。
DEFAULT_EPOCHS = 75
# 与远程基线 args.yaml 完全一致：warmup_epochs=1.0（不是 default.yaml 的 3.0）。
DEFAULT_WARMUP_EPOCHS = 1.0
DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
MODEL_DIR = "ultralytics/cfg/models/11"
MODEL_YAMLS = {
    "base": f"{MODEL_DIR}/yolo11.yaml",
    # MCSR 载体：与 base 逐层索引一致，仅头替换为 AdaptiveDefectDetect([nc, True, 16])。
    "mcsr": f"{MODEL_DIR}/yolo11n-adaptive-defect.yaml",
}
PRETRAINED = "yolo11n.pt"

# 内置消融实验（必须在 train.py 内运行）：--ablation <key> 在 train.py 内部串行训练每个配置并汇总。
# 每个条目只含 train_model 的关键字参数；name 为 run 目录名。对照 = 远程基线 0.72445（纯 stock）。
ABLATIONS: dict[str, list[dict]] = {
    # 只开 MCSR：结构侧单变量隔离。mcsr=True → yolo11n-adaptive-defect.yaml；sa_tal=False → stock TAL。
    "mcsr": [
        {"name": "mcsr", "mcsr": True, "sa_tal": False, "sa_size": 40.0},
    ],
    # 只开 SA-TAL：loss 分配侧单变量隔离。mcsr=False → base yaml；sa_tal=True → size-aware assigner。
    "sa_tal": [
        {"name": "sa_tal", "mcsr": False, "sa_tal": True, "sa_size": 40.0},
    ],
    # MCSR + SA-TAL 组合：达标主臂（两臂正交，预期可叠加）。
    "both": [
        {"name": "both", "mcsr": True, "sa_tal": True, "sa_size": 40.0},
    ],
}

# === 文件内开关：直接改这一行切实验，无需任何命令行参数 ===
#   RUN_ABLATION = "mcsr"   # 只开 MCSR（结构路由）
#   RUN_ABLATION = "sa_tal" # 只开 SA-TAL（尺寸感知分配）
#   RUN_ABLATION = "both"   # MCSR + SA-TAL 组合（默认，达标主臂）
#   RUN_ABLATION = None     # 关闭文件内开关，跑默认组合训练（纯 stock）
RUN_ABLATION: str | None = "both"


DESKTOP_DIR = Path(r"C:/Users/23223/Desktop/新论文数据保存/MCSR_SA-TAL消融实验_75ep")


def copy_run_to_desktop(run_dir: Path) -> Path:
    """把一次实验的完整 run 目录拷贝到桌面（含日志）。已存在则合并覆盖。"""
    DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    dst = DESKTOP_DIR / run_dir.name
    if dst.exists():
        shutil.rmtree(dst)  # 重跑同臂时整体替换，避免新旧文件混杂
    shutil.copytree(run_dir, dst)
    print(f"[desktop] 已拷贝 {run_dir.name} -> {dst}")
    return dst


def record_model_stats(model, imgsz: int = 640) -> str:
    """返回模型参数量(M)与计算量(GFLOPs)的格式化字符串（供日志/args 记录）。"""
    # verbose=True 才会返回 (n_l, n_p, n_g, flops)，且同时把 summary 行打进训练日志。
    n_l, n_p, n_g, flops = model.model.info(verbose=True, imgsz=imgsz)
    return f"layers={n_l} | params={n_p / 1e6:.3f}M ({n_p:,}) | gradients={n_g} | GFLOPs={flops:.3f}"


def select_model_yaml(mcsr: bool = False) -> str:
    """网络侧：默认纯 stock yolo11n；mcsr=True 用 yolo11n-adaptive-defect.yaml（AdaptiveDefectDetect/MCSR）。"""
    return MODEL_YAMLS["mcsr"] if mcsr else MODEL_YAMLS["base"]


def build_reproducible_model(seed: int = EXPERIMENT_SEED, mcsr: bool = False) -> YOLO:
    """固定随机状态后构建并加载 YOLO11n 预训练权重。"""
    init_seeds(seed, deterministic=True)
    return YOLO(select_model_yaml(mcsr=mcsr)).load(PRETRAINED)


def train_model(
    epochs: int | None = None,  # None -> DEFAULT_EPOCHS（单一事实来源）
    warmup_epochs: float = DEFAULT_WARMUP_EPOCHS,
    imgsz: int = 640,
    name: str | None = None,
    # --- 唯一保留的创新点开关 ---
    # MCSR（结构级）：True 时用 yolo11n-adaptive-defect.yaml（AdaptiveDefectDetect 头内路由）。
    # 恒等初始化（residual_gain=0）→ 初始输出逐位等于输入，预训练权重全量按名迁移。
    mcsr: bool = False,
    # SA-TAL（loss 级）：True 时 TAL assigner 对小 GT 框（对角线 < sa_size px）拓宽候选椭圆 +
    # 混合尺度不变 center-quality。False = stock TAL（byte-identical）。
    sa_tal: bool = False,
    sa_size: float = 40.0,
    # --- 训练环境 ---
    seed: int = 0,
    batch: int = 16,
    workers: int = 4,
    cache: bool = False,
    device: int = 0,
    exist_ok: bool = False,
):
    """在电池表面缺陷数据集上按开关训练指定配置的 YOLO11n。"""
    if epochs is None:
        epochs = DEFAULT_EPOCHS
    if name is None:
        name = "both" if (mcsr and sa_tal) else ("mcsr" if mcsr else ("sa_tal" if sa_tal else "baseline"))
    model = build_reproducible_model(seed=seed, mcsr=mcsr)
    # 记录本臂参数量/计算量（日志第一行即可见，用于论文对比表）
    stats = record_model_stats(model, imgsz=imgsz)
    print(f"[model_stats] {name}: {stats}")
    return model.train(
        data=DATASET_YAML,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        batch=batch,
        workers=workers,
        cache=cache,
        imgsz=imgsz,
        device=device,
        seed=seed,
        deterministic=True,
        save=True,
        resume=False,
        val=True,
        exist_ok=exist_ok,
        name=name,
        # 传入 train kwargs 以在 args.yaml 中记录（与远程基线 args.yaml 键集一致）。
        # mcsr/mcsr_hidden_dim 为记录用（结构由 yaml 决定）；sa_tal/sa_size 被 loss.py getattr 消费。
        mcsr=mcsr,
        mcsr_hidden_dim=16,
        sa_tal=sa_tal,
        sa_size=sa_size,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数（默认取 DEFAULT_EPOCHS）")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--name", type=str, default=None, help="运行名（默认由开关自动生成）")
    p.add_argument("--ablation", type=str, default=None,
                   choices=list(ABLATIONS.keys()),
                   help="在 train.py 内部串行跑内置消融实验（每配置训练后打印汇总）。可选: "
                        + ", ".join(ABLATIONS.keys()))
    # 普通模式创新点开关（消融模式下由 ABLATIONS 固定）
    p.add_argument("--mcsr", action=argparse.BooleanOptionalAction, default=False,
                   help="启用/关闭 MCSR（用 yolo11n-adaptive-defect.yaml）")
    p.add_argument("--sa-tal", action=argparse.BooleanOptionalAction, default=False,
                   help="启用/关闭 SA-TAL（尺寸感知 TAL 分配）")
    # 训练环境
    p.add_argument("--seed", type=int, default=EXPERIMENT_SEED,
                   help="随机种子（消融模式固定为 EXPERIMENT_SEED）")
    p.add_argument("--batch", type=int, default=16, help="batch size（消融模式固定为 16）")
    p.add_argument("--device", type=int, default=0, help="GPU 设备号")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False,
                   help="缓存图像到 RAM。注意：Windows spawn 下会把缓存 pickle 给每个 worker，6.8GB 缓存 x4 会 OOM，默认关闭")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker 数")
    p.add_argument("--exist-ok", action="store_true",
                   help="允许覆盖同名 run 目录（默认 False，存在则报错）")
    p.add_argument("--log-file", type=str, default=None,
                   help="本次训练的 stdout 日志文件路径（--ablation 模式完成后会随 run 一并拷到桌面）")
    return p.parse_args()


def run_ablation(
    key: str,
    epochs: int,
    batch: int,
    workers: int,
    cache: bool,
    device: int,
    seed: int,
    log_file: str | None = None,
) -> None:
    """在 train.py 内部串行训练 ABLATIONS[key] 里的每个配置，逐个打印 mAP50-95 汇总。"""
    entries = ABLATIONS[key]
    print(f"\n========== 消融实验 [{key}] 共 {len(entries)} 个配置 ==========")
    for i, over in enumerate(entries, 1):
        run_name = over["name"]
        # 非默认轮数（50）时给 run 名加 _e{epochs} 后缀，避免裁决轮数混淆。
        if epochs != DEFAULT_EPOCHS:
            run_name = f"{run_name}_e{epochs}"
        kwargs = {k: v for k, v in over.items() if k != "name"}
        print(
            f"\n---------- [{key}] {i}/{len(entries)} -> {run_name}  {kwargs} ----------")
        metrics = train_model(
            epochs=epochs,
            batch=batch,
            workers=workers,
            cache=cache,
            device=device,
            seed=seed,
            name=run_name,
            exist_ok=True,  # 消融重跑时原地覆盖同名 run，保持确定性对照
            **kwargs,
        )
        # DetMetrics: use results_dict
        m = metrics.results_dict.get("metrics/mAP50-95(B)", "?")
        # 逐类 mAP50-95（判定"每类都提高"需要）：names 是 {id: 类名}，box.maps 是逐类 mAP50-95
        try:
            names = metrics.names
            maps = metrics.box.maps
            per_class = ", ".join(
                f"{names[i]}: {float(maps[i]):.4f}" for i in range(len(maps)))
        except Exception:
            per_class = "?"
        print(f"[ablation] {run_name} 完成 -> mAP50-95(B) = {m}  | per-class: {per_class}")
        # 拷贝前 flush，确保 stdout 重定向日志在复制时已完整落盘。
        sys.stdout.flush()
        # 每次实验结束后：先把控制台日志并入 run 目录，再整体拷到桌面，
        # 确保 run 目录自含日志（即使之后手动重拷 run 目录也不会丢日志）。
        if log_file and Path(log_file).exists():
            run_path = Path("runs/detect") / run_name
            shutil.copy2(log_file, run_path / Path(log_file).name)
            print(f"[run] 已并入训练日志 {log_file} -> {run_path / Path(log_file).name}")
        dst = copy_run_to_desktop(Path("runs/detect") / run_name)
        print(f"[desktop] 已拷贝 {run_name}（含日志）-> {dst}")
        sys.stdout.flush()


def main() -> None:
    args = parse_args()
    if args.ablation is None and RUN_ABLATION is not None:
        args.ablation = RUN_ABLATION
    if args.ablation is not None:
        run_ablation(
            key=args.ablation,
            epochs=args.epochs,
            batch=16,  # 固定 batch（消融对照一致）
            workers=args.workers,
            cache=args.cache,
            device=0,
            seed=EXPERIMENT_SEED,
            log_file=args.log_file,
        )
        return
    train_model(
        epochs=args.epochs,
        imgsz=args.imgsz,
        name=args.name,
        mcsr=args.mcsr,
        sa_tal=args.sa_tal,
        seed=args.seed,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
