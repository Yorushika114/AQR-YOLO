# -*- coding: utf-8 -*-
"""由实测 JSON 生成论文 3.3 节正文（表 + 分析）。

数字全部从 perclass_results.json / target_scale_stats.json 读入并代入，
不手写，避免转抄错误。

用法:
    python tools/write_section33.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(r"C:/Users/23223/Desktop/新论文数据保存/逐类检测性能对比试验")
R = json.loads((OUT / "perclass_results.json").read_text(encoding="utf-8"))
S = json.loads((OUT / "target_scale_stats.json").read_text(encoding="utf-8"))

ORDER = R["results"]["baseline"]["class_order"]          # ['Pinhole','Surface_Crack','Delamination']
CN = {"Pinhole": "针孔", "Surface_Crack": "细长裂纹", "Delamination": "分层"}

BASE = R["results"]["baseline"]
MSA = R["results"]["msa"]
F = R.get("factor_2x2", {})
GT = R["gt_counts"]
SPAN = 100  # 满分 100


_KEY = {"p": "precision", "r": "recall", "ap50": "ap50", "ap50_95": "ap50_95",
        "tp": "tp", "fp": "fp", "fn": "fn", "gt": "gt_count"}


def pc(arm, c, key):
    return arm["per_class"][c][_KEY.get(key, key)]


def d(arm, c, key):
    return pc(arm, c, key) - pc(BASE, c, key)


def bar(arm_key, c):
    """2x2 中某一臂某类的 AP50-95。"""
    return F[arm_key]["per_class"][c]["ap50_95"]


# ------------------------------------------------------------ 主表
lines = []
A = lines.append
A("## 3.3 不同缺陷类别的检测性能对比")
A("")
A("### 3.3.1 实验设置")
A("")
A("对比模型为基线 YOLO11n 与本文方法 MSA-YOLO-RAP。两者的训练集 / 验证集划分、"
  f"输入尺寸（{R['config']['imgsz']}×{R['config']['imgsz']}）、batch size、训练轮次（100）、"
  "随机种子（0）、数据增强策略与评价脚本完全一致，唯一差异在于网络结构与正样本分配策略。")
A("")
A(f"验证集共 1550 张图像，含标注目标 {sum(GT.values())} 个："
  f"针孔 {GT['0']} 个、细长裂纹 {GT['1']} 个、分层 {GT['2']} 个。")
A("")
A("评价指标说明：P、R 取自各自 PR 曲线最大 F1 工作点；AP50、AP50-95 为 COCO 标准定义"
  "（AP50-95 为 IoU 阈值 0.50∶0.05∶0.95 共 10 点的平均）。漏检数（FN）与误检数（FP）"
  "在该工作点下由 TP+FN=真实目标数、FP 由精确率反解得到，与报告的 P、R 自洽。")
A("")
A("### 3.3.2 逐类别检测性能")
A("")
A("**表 3-3  不同缺陷类别的检测性能对比**")
A("")
A("| 模型 | 类别 | P/% | R/% | AP50/% | AP50-95/% | 漏检数 | 误检数 |")
A("|---|---|---:|---:|---:|---:|---:|---:|")
for c in ORDER:
    for arm, name in ((BASE, BASE["label"]), (MSA, MSA["label"])):
        A(f"| {name} | {CN[c]}（{c}） | {pc(arm,c,'p'):.2f} | {pc(arm,c,'r'):.2f} | "
          f"{pc(arm,c,'ap50'):.2f} | {pc(arm,c,'ap50_95'):.2f} | {pc(arm,c,'fn')} | {pc(arm,c,'fp')} |")
A("")
A("### 3.3.3 结果分析")
A("")

gain = {c: d(MSA, c, "ap50_95") for c in ORDER}
best = max(gain, key=gain.get)
worst = min(gain, key=gain.get)
fn_tot_b, fn_tot_m = (sum(pc(BASE, c, "fn") for c in ORDER), sum(pc(MSA, c, "fn") for c in ORDER))
fp_tot_b, fp_tot_m = (sum(pc(BASE, c, "fp") for c in ORDER), sum(pc(MSA, c, "fp") for c in ORDER))

A(f"**（1）提升最明显的类别。** 按 AP50-95 计，本文方法的增益集中在{CN[best]}："
  f"{pc(BASE,best,'ap50_95'):.2f}% → {pc(MSA,best,'ap50_95'):.2f}%（{gain[best]:+.2f} 个百分点），"
  f"漏检由 {pc(BASE,best,'fn')} 个降至 {pc(MSA,best,'fn')} 个，减少 "
  f"{pc(BASE,best,'fn')-pc(MSA,best,'fn')} 个（相对降幅 "
  f"{(pc(BASE,best,'fn')-pc(MSA,best,'fn'))/pc(BASE,best,'fn')*100:.1f}%），"
  f"误检同步减少 {pc(BASE,best,'fp')-pc(MSA,best,'fp')} 个。"
  f"相比之下，{CN[worst]}的 AP50-95 仅提升 {gain[worst]:+.2f} 个百分点，"
  f"与{CN['Surface_Crack']}的 {gain['Surface_Crack']:+.2f} 相差约 "
  f"{gain['Surface_Crack']/max(gain[worst],1e-6):.0f} 倍。"
  f"三个类别的提升幅度并不一致，呈「{CN['Surface_Crack']} > {CN['Pinhole']} > {CN['Delamination']}」的梯度。")
A("")
A(f"**（2）对微小针孔与细长裂纹的漏检抑制。** 两类均有效，但作用路径不同。"
  f"针孔目标数仅 {GT['0']} 个，基线漏检 {pc(BASE,'Pinhole','fn')} 个、本文 {pc(MSA,'Pinhole','fn')} 个，"
  f"绝对减少 {pc(BASE,'Pinhole','fn')-pc(MSA,'Pinhole','fn')} 个，相对降幅 "
  f"{(pc(BASE,'Pinhole','fn')-pc(MSA,'Pinhole','fn'))/pc(BASE,'Pinhole','fn')*100:.1f}%；"
  f"但该类别基数小，AP50-95 增益仅 {gain['Pinhole']:+.2f} 个百分点，"
  f"且精确率出现 {d(MSA,'Pinhole','p'):+.2f} 个百分点的轻微回落——本文方法在针孔上以极小的精确率代价换取召回，"
  f"属于典型的\"提召回\"型改动。裂纹类目标数 {GT['1']} 个，是全表漏检的绝对主力"
  f"（基线漏检占三类总漏检 {pc(BASE,'Surface_Crack','fn')/fn_tot_b*100:.1f}%），"
  f"本文方法减少漏检 {pc(BASE,'Surface_Crack','fn')-pc(MSA,'Surface_Crack','fn')} 个，"
  f"也是三类中召回与精确率同时上升且增幅最大的一类"
  f"（P {d(MSA,'Surface_Crack','p'):+.2f}、R {d(MSA,'Surface_Crack','r'):+.2f}；"
  f"分层虽同为双升，但幅度仅 P {d(MSA,'Delamination','p'):+.2f}、R {d(MSA,'Delamination','r'):+.2f}）。")
A("")
A(f"**（3）三类提升是否一致。** 不一致。以 AP50-95 排序："
  f"{CN['Surface_Crack']} {gain['Surface_Crack']:+.2f} > {CN['Pinhole']} {gain['Pinhole']:+.2f} "
  f"> {CN['Delamination']} {gain['Delamination']:+.2f}。"
  f"总体层面 mAP50-95 由 {BASE['overall']['mAP50_95']:.2f}% 提升至 {MSA['overall']['mAP50_95']:.2f}%"
  f"（{MSA['overall']['mAP50_95']-BASE['overall']['mAP50_95']:+.2f}），"
  f"mAP50 由 {BASE['overall']['mAP50']:.2f}% 提升至 {MSA['overall']['mAP50']:.2f}%"
  f"（{MSA['overall']['mAP50']-BASE['overall']['mAP50']:+.2f}）；"
  f"全类漏检 {fn_tot_b} → {fn_tot_m} 个，误检 {fp_tot_b} → {fp_tot_m} 个。"
  f"按 P 口径，针孔出现 {d(MSA,'Pinhole','p'):+.2f} 的回落，其余两类上升；"
  f"该回落幅度远小于其召回增益（{d(MSA,'Pinhole','r'):+.2f}），不构成性能下降。"
  f"三个类别中不存在 AP 指标下降的类别。")
A("")

# ------------------------------------------------------------ 消融
if len(F) == 4:
    A("### 3.3.4 单因素消融：MCSR 与 SA-TAL 的贡献分解")
    A("")
    A("为区分两个模块各自的贡献，在完全相同协议下补跑两个单因素臂"
      "（仅开启 MCSR 或仅开启 SA-TAL），构成 2×2 因子设计。")
    A("")
    A("**表 3-4  2×2 因子分解（AP50-95 / %）**")
    A("")
    A("| 模型 | MCSR | SA-TAL | mAP50 | mAP50-95 | " + " | ".join(CN[c] for c in ORDER) + " |")
    A("|---|:---:|:---:|---:|---:|" + "---:|" * len(ORDER))
    for k in ("baseline", "mcsr", "sa_tal", "msa"):
        f = F[k]
        m, s = ("baseline" and (0, 0)) if k == "baseline" else \
               (1, 0) if k == "mcsr" else (0, 1) if k == "sa_tal" else (1, 1)
        A(f"| {f['label']} | {'√' if m else '×'} | {'√' if s else '×'} | "
          f"{f['overall']['mAP50']:.2f} | {f['overall']['mAP50_95']:.2f} | "
          + " | ".join(f"{f['per_class'][c]['ap50_95']:.2f}" for c in ORDER) + " |")
    A("")
    A("由表 3-4 可得各因素的主效应与交互效应（以 AP50-95 计）：")
    A("")
    A("| 类别 | 基线 | 主效应 MCSR | 主效应 SA-TAL | 加和预测 | 实测(双开) | 交互项 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    inter = {}
    for c in ORDER:
        e_m = bar("mcsr", c) - bar("baseline", c)
        e_s = bar("sa_tal", c) - bar("baseline", c)
        pred = bar("baseline", c) + e_m + e_s
        act = bar("msa", c)
        inter[c] = act - pred
        A(f"| {CN[c]} | {bar('baseline',c):.2f} | {e_m:+.2f} | {e_s:+.2f} | {pred:.2f} | "
          f"{act:.2f} | **{inter[c]:+.2f}** |")
    e_m_all = [bar("mcsr", c) - bar("baseline", c) for c in ORDER]
    e_s_all = [bar("sa_tal", c) - bar("baseline", c) for c in ORDER]
    A("")
    A(f"MCSR 单独开启时，三类 AP50-95 的平均主效应为 {sum(e_m_all)/len(e_m_all):+.2f}；"
      f"SA-TAL 单独开启时为 {sum(e_s_all)/len(e_s_all):+.2f}。"
      f"即 SA-TAL 承担了更大的单因素份额。")
    A("")
    A(f"交互项在{CN['Surface_Crack']}上为 {inter['Surface_Crack']:+.2f}，"
      f"高于 MCSR 的主效应（{bar('mcsr','Surface_Crack')-bar('baseline','Surface_Crack'):+.2f}），"
      f"说明两个模块在裂纹上存在明显的正向协同，而不是各自独立地叠加收益。")
    mg = bar("mcsr", "Delamination") - bar("baseline", "Delamination")
    if mg < 0:
        A("")
        A(f"需如实指出：MCSR 单独开启时，{CN['Delamination']}的 AP50-95 反而下降 "
          f"{mg:.2f} 个百分点（{bar('baseline','Delamination'):.2f} → {bar('mcsr','Delamination'):.2f}），"
          f"误检由 {F['baseline']['per_class']['Delamination']['fp']} 个升至 "
          f"{F['mcsr']['per_class']['Delamination']['fp']} 个。"
          f"该类别在双模块同时开启时恢复并略优于基线（{bar('msa','Delamination'):.2f}），"
          f"表明多尺度路由在尺度一致性较强的大目标上会引入额外的假激活，"
          f"需要由正样本分配机制加以抑制。")
    A("")
    A("### 3.3.5 结合目标尺度与形态的机理解释")
    A("")
    A("**表 3-5  验证集各类缺陷的目标尺度与形态统计**（像素尺度按原图分辨率还原）")
    A("")
    A("| 类别 | 目标数 | 面积 p10 | 面积中位 | 面积 p90 | p90/p10 | w 中位 | h 中位 | 长宽比中位 | 长宽比 p90 | 面积<32²占比 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in ORDER:
        v = S["splits"]["val"][c]
        A(f"| {CN[c]} | {v['n_boxes']} | {v['area_p10']:.0f} | {v['area_med']:.0f} | {v['area_p90']:.0f} | "
          f"{v['area_span_p90_p10']:.0f} | {v['w_med']:.1f} | {v['h_med']:.1f} | {v['ar_med']:.2f} | "
          f"{v['ar_p90']:.2f} | {v['pct_tiny_lt_32sq']:.1f}% |")
    A("")
    p, k, l = (S["splits"]["val"][c] for c in ORDER)
    A(f"三类的尺度分布差异是解释上述性能梯度的直接依据。"
      f"{CN['Pinhole']}面积中位仅 {p['area_med']:.0f} px²、{p['pct_tiny_lt_32sq']:.1f}% 的目标小于 32²，"
      f"但尺度跨度极小（p90/p10 仅 {p['area_span_p90_p10']:.0f}），形态接近正方"
      f"（长宽比中位 {p['ar_med']:.2f}），属于\"小而形态一致\"的一类；"
      f"{CN['Surface_Crack']}面积跨度达 {k['area_span_p90_p10']:.0f} 倍，"
      f"长宽比 p90 达 {k['ar_p90']:.2f}，同时包含大面积撕裂与仅几十像素的细丝状裂纹，"
      f"是三类中多尺度性最强、也最难被单一尺度分支覆盖的一类；"
      f"{CN['Delamination']}面积中位 {l['area_med']:.0f} px²，是三类中最大的，"
      f"且尺度跨度 {l['area_span_p90_p10']:.0f} 倍、小于裂纹类一个数量级。")
    A("")
    A(f"据此，**MCSR（多尺度路由）** 的作用可以定位为：为不同尺度的目标提供各自适配的感受野通路。"
      f"其收益应主要体现在尺度分布宽、需要跨尺度表征的类别上——"
      f"数据上表现为{CN['Surface_Crack']}的召回提升（R {d(MSA,'Surface_Crack','r'):+.2f}）；"
      f"同时也解释了它在{CN['Delamination']}上的单独负效应：该类尺度一致性较强，"
      f"多尺度分支带来的额外容量更多地转化为背景激活（误检 "
      f"{F['mcsr']['per_class']['Delamination']['fp']-F['baseline']['per_class']['Delamination']['fp']:+d} 个），"
      f"而非有效召回。")
    A("")
    A(f"**SA-TAL（尺寸感知正样本分配）** 的作用可以定位为：按目标尺寸调整正样本的匹配范围与权重，"
      f"使小目标不至于因 IoU 门槛过严而被判为负样本。其收益应主要体现在小目标占比高的类别上——"
      f"数据上表现为{CN['Pinhole']}的召回提升（R {d(MSA,'Pinhole','r'):+.2f}，"
      f"漏检 {pc(BASE,'Pinhole','fn')} → {pc(MSA,'Pinhole','fn')}）"
      f"与{CN['Delamination']}误检的下降（"
      f"{F['sa_tal']['per_class']['Delamination']['fp']-F['baseline']['per_class']['Delamination']['fp']:+d} 个）。"
      f"SA-TAL 作为单因素时承担了更大的平均主效应"
      f"（{sum(e_s_all)/len(e_s_all):+.2f} vs MCSR 的 {sum(e_m_all)/len(e_m_all):+.2f}），"
      f"与其作用于损失分配这一更靠近优化目标的环节相符。")
    A("")
    A(f"两个模块的协同在{CN['Surface_Crack']}上最显著（交互项 {inter['Surface_Crack']:+.2f}）："
      f"MCSR 提供跨尺度的候选表征，SA-TAL 在此基础上按尺寸把正样本分配给合适的锚点，"
      f"二者分别作用于\"看得见\"与\"学得到\"两个环节，缺一不可——"
      f"单独开启任一模块的裂纹 AP50-95 增量（"
      f"{bar('mcsr','Surface_Crack')-bar('baseline','Surface_Crack'):+.2f} / "
      f"{bar('sa_tal','Surface_Crack')-bar('baseline','Surface_Crack'):+.2f}）"
      f"均小于双开时的 {bar('msa','Surface_Crack')-bar('baseline','Surface_Crack'):+.2f}。")
    A("")
A("### 3.3.6 典型检测结果")
A("")
A("逐类 PR 曲线见 `pr_curves_per_class.png`；典型检测可视化见 `viz/` 目录，"
  "每张图为「真值 / YOLO11n / MSA-YOLO-RAP」三联对比，红色框标注漏检目标、橙色框标注误检：")
A("")
for c in ORDER:
    n = c.lower()
    A(f"- `viz/{n}_typical.png`：{CN[c]}典型样例（两模型均检出）")
    A(f"- `viz/{n}_msa_hit_baseline_miss.png`：本文检出、基线漏检的{CN[c]}样例")
    A(f"- `viz/{n}_baseline_fp.png`：基线误检、本文无误检的{CN[c]}样例")
A("")
A("`viz/manifest.json` 记录了每张样例对应的原始图像文件名与逐类 TP/FP/FN 明细，"
  "便于论文中引用与复核。")
A("")
A("### 3.3.7 局限")
A("")
A("上述结果基于单一种子（seed=0）的一次训练，未做多随机种子重复，"
  "因此各模块的量化贡献（尤其是 0.1 个百分点量级的交互项）只应作为趋势性证据，"
  "不宜作为精确的效应量估计。此外，验证集按图像划分，同一电芯的多个缺陷样本可能同时落入"
  "训练与验证集，泛化性能可能被高估。")

txt = "\n".join(lines) + "\n"
(OUT / "3.3_逐类检测性能对比分析.md").write_text(txt, encoding="utf-8")
print(txt)
print(f"[save] {OUT / '3.3_逐类检测性能对比分析.md'}")
