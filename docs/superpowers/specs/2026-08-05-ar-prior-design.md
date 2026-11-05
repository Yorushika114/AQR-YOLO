# AR-Prior:宽高比几何先验(Aspect-Ratio Geometric Prior)设计

日期:2026-08-05
状态:已获用户批准(方案 A + 删 scol/eiou/mcsr)

## 1. 背景与目标

电池表面缺陷检测(Pinhole / Surface_Crack / Delamination,3 类),YOLO11n 基底(yolo11n.pt),imgsz 640。

50 轮裁决消融(4 臂)已出终审:全部现有创新点证伪 —— 无臂达到 +0.03~0.05 目标;最强的 SA-TAL 仅 +0.0081,且 scol/eiou 与 SA-TAL 小目标收益高度重叠,full 组合对 Delamination 交互为负。裁决基线:sa_on_e50 = 0.7957(0.8507 / 0.6677 / 0.8687)。

**本设计目标**:保留 SA-TAL 为基底,叠加一个新的数学/数据结构方向创新点 —— **宽高比几何先验(AR-Prior)**,经 50 轮消融验证能否在基底上带来 **+0.03~0.05 且三类 mAP50-95 全涨**。若再证伪,则执行兜底决策树:去掉 sa_tal → 纯 stock yolo11n。

## 2. 问题与数据证据(SA-TAL 盲区)

SA-TAL 只对 **diag < 40px** 的小框生效(`s_small = clamp(1 - diag/sa_size)`)。但细长裂纹往往是 **"长"而非"小"**(diag ≥ 40),完全落在盲区。

训练集 37602 个 GT 框的分布(imgsz 640 像素):

| 类别 | n | AR 中位/p90 | AR≥3 占比 | diag≥40 且 AR≥3(SA-TAL 盲区) | diag<40 且 AR≥3 |
|---|---|---|---|---|---|
| Pinhole | 3176 | 1.3 / 2.0 | 0.4% | 0.4%(12 框) | 0 |
| **Surface_Crack** | **21053** | 1.9 / 4.7 | 24.1% | **20.0%(4211 框)** | 4.2% |
| Delamination | 13373 | 1.6 / 4.0 | 17.6% | 16.7%(2235 框) | 1.4% |
| 全部 | 37602 | — | 20.0% | **17.2%(6446 框)** | — |

### 细长框的饿死机制(与 SA-TAL 攻的小框机制同源)

1. **IoU 封顶**:锚点框近似方形(约 stride 大小),与 200×4 的细条重叠 → 最佳 IoU ≤ ~0.5,现实更差;`align_metric = cls^0.5 · IoU^6` 塌缩到 ~1e-3,target_scores(软目标)≈ 最佳 IoU → cls 学不会开火。
2. **SA-TAL 的各向同性距离是错的**:`d_norm = 欧氏距离/diag`,对 200×4 裂纹,偏裂纹线 4px 与沿裂纹走 50px 得到相同惩罚(都 ÷200)—— 它无法区分「离开了裂纹」与「还在裂纹上」。数学上需要一个各向异性(椭圆)度量。
3. **候选稀疏**:4px 细带内的锚点行可能落在网格线之间 → 框内候选为 0,连被 top-k 选中的机会都没有。

## 3. 删除范围(已确认)

| 模块 | 删除内容 |
|---|---|
| scol | loss.py:576-578 三超参、682-693 size_weight 块、700-701 乘权;default.yaml 的 scol/scol_alpha/scol_size;train.py 的 --scol/--scol-alpha/--scol-size 及 ABLATIONS 相关键 |
| eiou | loss.py:617-618 eiou_enabled + BboxLoss 传参;BboxLoss:113-140 的 eiou 分支精简为 CIoU-only(RotatedBboxLoss 恒默认 False,不受影响);default.yaml 的 eiou;train.py 的 --eiou |
| mcsr | head.py 的 `AdaptiveDefectDetect` 类删除、恢复 stock `Detect`;battery_next.py 整文件删除(含 MorphologyConditionedScaleRouter);yolo11n-adaptive-defect.yaml 删除;loss.py:571-572 的 `set_adaptive_modules_enabled` 调用删除;default.yaml 的 mcsr/mcsr_hidden_dim;train.py 的 mcsr 开关与 MODEL_YAMLS["mcsr"] |

**保留**:SA-TAL(tal.py size_aware 机制 + loss.py sa_tal/sa_size 接线)、consensus 推理重排、class_weights、head.py 的 cv3_iou 网络结构(被 OBB/Pose/Segment deepcopy 共享)、metrics.py 的 EIoU 共享原语(通用工具,不再被本项目调用)。

**关键事实**:sa_on_e50=0.7957 的对照臂本就是 `mcsr=False`,故删除 mcsr/scol/eiou **不改变基底参照**。新消融的 sa_base 臂目标即复现 0.7957,单变量隔离干净。

## 4. 新模块:AR-Prior(椭圆分配质量)

### 4.1 数学

对 GT 框(中心 gx,gy;宽 w,高 h;半轴 a=w/2, b=h/2)与锚点(ax,ay):

```
椭圆马氏距离(各向异性):  d_ellip = hypot((ax-gx)/w, (ay-gy)/h)     (单位化为整宽/整高)
椭圆质量:                ellip_q = exp(-d_ellip²)                   (1=中心, 0.37=内切椭圆边缘)
```

SA-TAL 的 `iso_q = exp(-d_norm²)`, `d_norm = hypot(Δx,Δy)/diag` 是 d_ellip 在**方框**下的特例;AR-Prior 把它严格推广为 GT 协方差诱导的**各向异性马氏距离**。200×4 裂纹上:偏 4px(整条宽度)→ `(4/4)²=1` → ellip_q=0.37;沿裂纹 50px → `(50/200)²=0.0625` → 0.94。在裂纹线上的锚点拿到 ≈1 的软目标 —— 线段的真实几何。

### 4.2 伸长门控(与 SA-TAL 小框门控并行)

```
ar      = max(w,h) / max(min(w,h), eps)
s_elong = clamp((ar - ar_min) / (ar_max - ar_min), 0, 1)     # ar_min=2.0, ar_max=4.0
s_small = clamp(1.0 - diag / sa_size, 0, 1)                  # SA-TAL 原式,保留
s       = clamp(s_small + s_elong, 0, 1)                     # 尺寸 ∪ 伸长,统一混合强度
```

- 方框(ar≈1):s_elong=0 → 质量退化为 iso_q(SA-TAL 行为逐位保留)。
- 细长框:s_elong→1 → ellip_q 主导,即使 diag 巨大(盲区被激活)。
- 小且细长:s 双份贡献,ellip_q 胜出(细薄维度是绑定约束)。

### 4.3 质量 pow-blend(与 SA-TAL 同一结构)

```
q       = where(s_elong > 0, ellip_q, iso_q)      # 有任何伸长即用椭圆;纯方框用 SA-TAL 原式
blended = (overlaps + eps)^(1-s) · (q + eps)^s    # overlaps=IoU
overlaps = where(s > 0, blended, overlaps)
```

`s_elong=0` 时逐位等同当前 SA-TAL 基底(s=0 → overlaps 不变)。soft target(`target_scores ∝ overlaps`)对裂纹锚点给出真实可学值。

### 4.4 候选拓宽(细长框沿主轴椭圆带,与 SA-TAL 小框圆盘并行)

```
mask_in_gts = mask_in_gts.bool() | (d_ellip < ar_radius) & (s_elong > 0.25)     # ar_radius=1.5
```

解决 4px 细带锚点行落在网格线之间导致的候选为 0;椭圆带沿主轴延伸 1.5·a、垂直方向仅 1.5·b,不污染邻接目标。

### 4.5 开关与接线

- **tal.py** `TaskAlignedAssigner.__init__` 新增 `ar_prior: bool = False, ar_min=2.0, ar_max=4.0, ar_radius=1.5`;`get_pos_mask` 加椭圆带拓宽;`get_box_metrics` 加伸长门控 + 椭圆质量。
- **loss.py** `v8DetectionLoss.__init__` 从 hyp 读 `ar_prior/ar_min/ar_max/ar_radius` 传给 assigner;`ar_prior=False` 时与当前 SA-TAL 基底逐位一致。
- **default.yaml** 加 `ar_prior / ar_min / ar_max / ar_radius` 四键;删 `scol/scol_alpha/scol_size/eiou/mcsr/mcsr_hidden_dim` 六键。
- **train.py**:
  - ABLATIONS 缩为两臂:
    ```python
    "sa_base": [{"name": "sa_base", "sa_tal": True, "sa_size": 40.0, "ar_prior": False}],
    "sa_ar":   [{"name": "sa_ar",   "sa_tal": True, "sa_size": 40.0, "ar_prior": True,
                  "ar_min": 2.0, "ar_max": 4.0, "ar_radius": 1.5}],
    ```
  - `RUN_ABLATION = "sa_ar"`;删 mcsr/scol/eiou 全部开关;`MODEL_YAMLS` 只剩 base(yolo11.yaml);`select_model_yaml` 恒返回 yolo11.yaml。
  - 保留 sa_tal/sa_size 开关;新加 --ar-prior/--ar-min/--ar-max/--ar-radius。

## 5. 验证协议(50 轮裁决)

1. **静态冒烟**:`python -m py_compile` 全部改动文件;`python train.py --help`;stock Detect 构建冒烟(`YOLO('ultralytics/cfg/models/11/yolo11.yaml')`);确认 ABLATIONS 只剩两臂。
2. **AR 门控触发校验**:单轮训练日志中统计 `s_elong>0` 的 GT 占比(期望 ≈17%)、椭圆带实际捞到的候选数>0,确认模块真在数据上激活。
3. **5 轮冒烟**:sa_base 与 sa_ar 各 5 轮,loss 正常、无 NaN、run 名带 _e5。
4. **50 轮消融**(640 / batch16 / seed0 / warmup1.0,串行,`--ablation sa_ar --epochs 50` 或显式逐臂):`sa_base_e50`(期望复现 ~0.7957)→ `sa_ar_e50`。
5. **裁决标准**:`sa_ar - sa_base ∈ [+0.03, +0.05]` **且三类 Pinhole / Surface_Crack / Delamination 的 mAP50-95 全部提高** → 通过;否则执行兜底决策树(去掉 sa_tal,纯 stock yolo11n 验收,即"mscr/scol/eiou/satal 全部去掉")。
6. **可视化**:`_live_plot.py` 颜色映射与 `live.html` 图例更新为 sa_base/sa_ar 两臂。

## 6. 风险与兜底

- **唯一赌注**:攻击 6446 框盲区(占全部 GT 17.2%)能否换来 +0.03。SA-TAL 攻 3176 框的 Pinhole 得 +0.0081;新模块目标群体大 2 倍、正落在最弱类 Surface_Crack 上、机制同源,是当前证据下最有希望的方向。若达不到 +0.03~0.05,按兜底决策树降级。
- **不误删**:metrics.py EIoU 原语(共享)、cv3_iou 网络结构(deepcopy 共享)、distill_model.py(仓库原生)、default.yaml 的 distill_model/dis/cls_pw(engine 直接 args 读取)全部保留。
- **回退路径**:AR-Prior 开关默认 False,删除操作与开关解耦;任何时刻可回到 SA-TAL 基底。
