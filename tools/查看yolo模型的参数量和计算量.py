from ultralytics import YOLO
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
import torch
import torch.nn as nn


# class C3k2_v2(nn.Module):
#     """适配剪枝的C3k2版本 - 显式路径版"""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)
#         self.cv0 = Conv(c1, self.c, 1, 1)
#         self.cv1 = Conv(c1, self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(
#             (3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         # 1. 显式定义前两个分支
#         out0 = self.cv0(x)
#         out1 = self.cv1(x)

#         # 2. 显式迭代，不使用复杂的列表推导式
#         outputs = [out0, out1]
#         current = out1
#         for layer in self.m:
#             current = layer(current)  # 每一层的结果清晰传递
#             outputs.append(current)

#         # 3. 静态拼接
#         return self.cv2(torch.cat(outputs, 1))


def analyze_yolo_model_ultralytics(model_path):
    """使用Ultralytics内置方法分析模型"""
    print("加载YOLO模型...")

    # 加载模型
    model = YOLO(model_path)

    # 方法1：使用YOLO内置的info方法
    print("\n" + "=" * 80)
    print("Ultralytics模型信息")
    print("=" * 80)

    # 获取模型信息
    info = model.info()
    print(f"模型信息: {info}")

    # 方法2：手动计算参数
    print("\n" + "=" * 80)
    print("详细参数统计")
    print("=" * 80)

    pytorch_model = model.model
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    trainable_params = sum(p.numel()
                           for p in pytorch_model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    print(f"模型大小: {total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"模型大小: {total_params * 2 / (1024**2):.2f} MB (float16)")

    return total_params


# 使用
analyze_yolo_model_ultralytics(
    r"C:\Users\23223\Desktop\新论文数据保存\对比实验数据\paper09_eim_fixed_75ep_v2\weights\best.pt")
