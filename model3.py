import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple
from src.SOLO import surrogate  

# ✅ 랜덤 시드 고정 (재현성 확보)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# ✅ LIF 뉴런 정의 (스파이킹 뉴런)
class LIF_Node(nn.Module):
    def __init__(self, LIF_tau: float, surrogate_function=surrogate.HeavisideBoxcarCall):
        super().__init__()
        self.LIF_decay = torch.sigmoid(torch.tensor(LIF_tau))
        self.surrogate_function = surrogate_function

    def forward(self, LIF_U, S_before, I_in):
        LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

# ✅ 기본적인 Conv 레이어 (BN 포함)
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ✅ YOLOv3-tiny에서 사용된 Conv_1
class Conv_1(nn.Module):
    def __init__(self, c1, c2, k, s, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p else k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))

# ✅ Residual Block (BasicBlock_2)
class BasicBlock_2(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s)
        self.conv2 = Conv(c2, c2, k, 1)
        self.shortcut = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, s)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)

# ✅ 업샘플링 (YOLO Head에서 사용됨)
class Sample(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.up(x)

# ✅ Feature Concatenation (YOLO에서 사용됨)
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)

# ✅ YOLO Detection Head
class Detect(nn.Module):
    def __init__(self, c_in, num_classes, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(c_in, num_anchors * (num_classes + 5), 1, 1)

    def forward(self, x):
        return self.conv(x)

# ✅ EMS-YOLO (YOLOv3-Tiny + SNN)
class EMS_YOLO(nn.Module):
    def __init__(self, num_classes=80, num_steps=5, init_LIF_tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps

        # ✅ Backbone (YOLOv3-tiny)
        self.conv1 = Conv_1(3, 64, 7, 2)  # Conv Layer
        self.LIF1 = LIF_Node(LIF_tau=init_LIF_tau)

        self.block1 = BasicBlock_2(64, 64, 3, 2)  
        self.block2 = BasicBlock_2(64, 64, 3, 1)  
        self.block3 = BasicBlock_2(64, 128, 3, 2)  
        self.block4 = BasicBlock_2(128, 128, 3, 1)  
        self.block5 = BasicBlock_2(128, 256, 3, 2)  
        self.block6 = BasicBlock_2(256, 256, 3, 1)  
        self.block7 = BasicBlock_2(256, 512, 3, 2)  
        self.block8 = BasicBlock_2(512, 512, 3, 1)  

        # ✅ Detection Head (YOLOv3-tiny)
        self.block9 = BasicBlock_2(512, 256, 3, 1)  
        self.conv10 = Conv(256, 512, 3, 1)  # P5/32-large

        self.upsample = Sample(scale_factor=2)  # 업샘플링
        self.concat = Concat(dim=1)  # P4 + P5 결합
        self.conv11 = Conv(384, 256, 3, 1)  # P4/16-medium

        # ✅ YOLO Detection Layer
        self.detect = Detect(256, num_classes, 3)  

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # ✅ 초기 Membrane Potential & Spikes 설정 (0.5로 채우기)
        c1_mem = c1_spike = torch.full((batch_size, 64, x.size(2), x.size(3)), 0.5, device=device)

        # ✅ Backbone
        c1_mem, c1_spike = self.LIF1(c1_mem, c1_spike, self.conv1(x))  
        x = self.block1(c1_spike)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        # ✅ Detection Head
        p5 = self.block9(x)  
        p5 = self.conv10(p5)  

        p4 = self.upsample(x)  
        fused = self.concat([p4, p5])  
        fused = self.conv11(fused)  

        # YOLO Detection
        detect = self.detect(fused)

        return detect
