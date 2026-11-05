# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Battery defect-specific network modules (创新点：MorphologyConditionedScaleRouter, 结构/网络侧).

MCSR 是"形状条件多尺度路由"模块，挂载在检测头（AdaptiveDefectDetect）前，对 P3/P4/P5
三尺度特征做跨尺度信息路由：
- 每张特征图的"形态学描述子"（每尺度 mean/std + 水平/垂直梯度能量，共 4 维/尺度）作为条件；
- 一个线性路由网络输出 num_scales×num_scales 的 softmax 路由矩阵（target 行 / source 列）；
- 对每个 target 尺度，把各 source 尺度特征经 1×1 投影对齐到 target 通道并上采样到 target 空间，
  按路由权重加权合成"路由消息"，再经 1×1 卷积、乘可学残差门控 residual_gain 后加回 target。

恒等初始化：residual_gain 初始为 0 → 模块初始输出 == 输入（纯附加、零扰动），
预训练权重可全量迁移；fuse conv normal_(0, 1e-3) 非零初始化保证首个 batch 即有梯度、不死锁。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyConditionedScaleRouter(nn.Module):
    """Route a residual message across detection scales from image-specific morphology statistics.

    结构创新（网络侧）：
    - ``source_projection``：num_scales×num_scales 的 1×1 Conv（source→target 全连接，bias=False），
      把每个 source 尺度的特征投影到每个 target 尺度的通道数。
    - ``output``：每 target 尺度一个 1×1 Conv（normal_(0,0.001) + zeros_ bias 近恒等初始化）。
    - ``residual_gain``：nn.Parameter(torch.zeros(num_scales))，可学残差门控（初始 0 → 恒等）。
    - ``router``：Linear(descriptor_dim→hidden)→SiLU→Linear(hidden→num_scales²)，输出 softmax 路由矩阵。

    描述子：每尺度拼接 [mean, std(unbiased=False), 水平梯度能量, 垂直梯度能量]（全局标量），
    共 ``num_scales×4`` 维。该统计描述电池表面的形态学结构（对比度/纹理方向性），
    路由器据此决定各尺度间如何交换信息——小尺度（P3）聚焦微小缺陷细节，大尺度（P5）提供上下文。
    """

    def __init__(self, channels, hidden_dim: int = 16):
        """Initialize the scale router.

        Args:
            channels (sequence of int): per-scale input channel counts (e.g. (64, 128, 256) for P3/P4/P5).
            hidden_dim (int): hidden width of the routing MLP.
        """
        super().__init__()
        if len(channels) < 2 or any(c < 1 for c in channels):
            raise ValueError("MCSR requires at least two positive feature channels")
        if hidden_dim < 1:
            raise ValueError("MCSR hidden_dim must be positive")
        self.channels = tuple(int(c) for c in channels)
        self.num_scales = len(self.channels)
        self.descriptor_dim = self.num_scales * 4
        self.router = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_scales * self.num_scales),
        )
        self.source_projection = nn.ModuleList(
            nn.ModuleList(nn.Conv2d(source, target, 1, bias=False) for source in channels) for target in channels
        )
        self.output = nn.ModuleList([nn.Conv2d(c, c, 1) for c in channels])
        self.residual_gain = nn.Parameter(torch.zeros(self.num_scales))
        for p in self.output:
            nn.init.normal_(p.weight, mean=0.0, std=0.001)
            nn.init.zeros_(p.bias)

    def _descriptor(self, feature: torch.Tensor) -> torch.Tensor:
        """Per-scale morphology statistics -> (b, 4): mean / std / horizontal / vertical gradient energy."""
        horizontal = (feature[..., 1:] - feature[..., :-1]).abs().mean(dim=(1, 2, 3))
        vertical = (feature[..., 1:, :] - feature[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        return torch.stack(
            [
                feature.mean(dim=(1, 2, 3)),
                feature.std(dim=(1, 2, 3), unbiased=False),
                horizontal,
                vertical,
            ],
            dim=1,
        )

    def describe(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate per-scale descriptors -> (b, num_scales*4)."""
        if len(features) != self.num_scales:
            raise ValueError(f"MCSR expected {self.num_scales} features, got {len(features)}")
        return torch.cat([self._descriptor(f) for f in features], dim=1)

    def forward(self, features: list[torch.Tensor], descriptor: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Route cross-scale residual messages and return the routed feature list.

        Args:
            features (list of torch.Tensor): per-scale feature maps (B, C_i, H_i, W_i).
            descriptor (torch.Tensor | None): precomputed descriptor (b, num_scales*4); None -> computed here.

        Returns:
            list of torch.Tensor: same shapes as ``features``; ``out[i] = feat[i] + residual_gain[i]*fuse_i(routed)``.
        """
        if descriptor is None:
            descriptor = self.describe(features)
        routing = self.router(descriptor).view(-1, self.num_scales, self.num_scales).softmax(dim=-1)
        outputs = []
        for target_index, target in enumerate(features):
            messages = []
            for source_index, source in enumerate(features):
                message = self.source_projection[target_index][source_index](source)
                if message.shape[-2:] != target.shape[-2:]:
                    message = F.interpolate(message, size=target.shape[-2:], mode="nearest")
                messages.append(message)
            stacked = torch.stack(messages, dim=1)  # (b, num_scales, C, H, W)
            weights = routing[:, target_index].view(-1, self.num_scales, 1, 1, 1)  # (b, num_scales, 1, 1, 1)
            routed = (stacked * weights).sum(dim=1)
            outputs.append(target + self.residual_gain[target_index] * self.output[target_index](routed))
        return outputs
