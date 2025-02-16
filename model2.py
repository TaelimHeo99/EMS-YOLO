import torch
import torch.nn as nn
import torch.nn.functional as F
from src.SOLO import surrogate

# Define LIF Node
class LIF_Node(nn.Module):
    def __init__(self, LIF_tau: float, surrogate_function=surrogate.HeavisideBoxcarCall):
        super().__init__()
        self.LIF_decay = torch.sigmoid(torch.tensor(LIF_tau))
        self.surrogate_function = surrogate_function

    def forward(self, LIF_U, S_before, I_in):
        LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

# Define EMS Blocks
class LCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lif = LIF_Node(LIF_tau=1.0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mem, spike):
        mem, spike = self.lif(mem, spike, self.bn(self.conv(x)))
        return mem, spike

class EMS_Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lcb1 = LCB(in_channels, out_channels)
        self.lcb2 = LCB(out_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, mem1, spike1, mem2, spike2):
        mem1, spike1 = self.lcb1(x, mem1, spike1)
        mem2, spike2 = self.lcb2(spike1, mem2, spike2)
        return self.maxpool(spike2), mem1, spike1, mem2, spike2

class EMS_Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lcb1 = LCB(in_channels, out_channels)
        self.lcb2 = LCB(out_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, mem1, spike1, mem2, spike2):
        mem1, spike1 = self.lcb1(x, mem1, spike1)
        mem2, spike2 = self.lcb2(spike1, mem2, spike2)
        pooled = self.maxpool(spike2)
        return torch.cat([pooled, spike2], dim=1), mem1, spike1, mem2, spike2

# Define the EMS-YOLO Model with Mem & Spike Tracking
class EMS_YOLO(nn.Module):
    def __init__(self, num_steps=5, num_classes=80):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        
        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.LIF1 = LIF_Node(LIF_tau=1.0)
        
        # Backbone Layers
        self.block1 = EMS_Block1(64, 128)
        self.block2 = EMS_Block2(128, 256)
        self.block3 = EMS_Block1(256, 512)
        
        # Feature Fusion
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat_conv = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1)
        
        # Detection Head
        self.detect_conv = nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initial Convolution
        mem1, spike1 = torch.full_like(x, 0.5), torch.zeros_like(x)
        mem1, spike1 = self.LIF1(mem1, spike1, self.bn1(self.conv1(x)))
        
        # Backbone Processing
        mem2, spike2, mem3, spike3 = torch.full_like(mem1, 0.5), torch.zeros_like(mem1), torch.full_like(mem1, 0.5), torch.zeros_like(mem1)
        out1, mem2, spike2, mem3, spike3 = self.block1(spike1, mem2, spike2, mem3, spike3)
        
        mem4, spike4, mem5, spike5 = torch.full_like(mem3, 0.5), torch.zeros_like(mem3), torch.full_like(mem3, 0.5), torch.zeros_like(mem3)
        out2, mem4, spike4, mem5, spike5 = self.block2(out1, mem4, spike4, mem5, spike5)
        
        mem6, spike6, mem7, spike7 = torch.full_like(mem5, 0.5), torch.zeros_like(mem5), torch.full_like(mem5, 0.5), torch.zeros_like(mem5)
        out3, mem6, spike6, mem7, spike7 = self.block3(out2, mem6, spike6, mem7, spike7)
        
        # Feature Fusion
        fused_feature = torch.cat([self.upsample(out1), out3], dim=1)
        fused_feature = self.concat_conv(fused_feature)
        
        # Detection Head
        mem_detect, spike_detect = torch.full_like(fused_feature, 0.5), torch.zeros_like(fused_feature)
        mem_detect, spike_detect = self.LIF1(mem_detect, spike_detect, self.detect_conv(fused_feature))
        
        return spike_detect  # YOLO Output

# Model Initialization
model = EMS_YOLO(num_steps=5, num_classes=80)
print(model)
