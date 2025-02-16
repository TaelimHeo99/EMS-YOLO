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
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.LIF2 = LIF_Node(LIF_tau=1.0)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.LIF3 = LIF_Node(LIF_tau=1.0)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.LIF4 = LIF_Node(LIF_tau=1.0)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.LIF5 = LIF_Node(LIF_tau=1.0)
        
        # Feature Fusion
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat_conv = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        # Detection Head
        self.detect_conv = nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initial Convolution
        c1_mem = c1_spike = torch.full_like(x, 0.5)
        c1_mem, c1_spike = self.LIF1(c1_mem, c1_spike, self.bn1(self.conv1(x)))
        
        # Backbone Processing
        c2_mem = c2_spike = torch.full_like(c1_mem, 0.5)
        c2_mem, c2_spike = self.LIF2(c2_mem, c2_spike, self.bn2(self.conv2(c1_spike)))
        
        c3_mem = c3_spike = torch.full_like(c2_mem, 0.5)
        c3_mem, c3_spike = self.LIF3(c3_mem, c3_spike, self.bn3(self.conv3(c2_spike)))
        
        c4_mem = c4_spike = torch.full_like(c3_mem, 0.5)
        c4_mem, c4_spike = self.LIF4(c4_mem, c4_spike, self.bn4(self.conv4(c3_spike)))
        
        c5_mem = c5_spike = torch.full_like(c4_mem, 0.5)
        c5_mem, c5_spike = self.LIF5(c5_mem, c5_spike, self.bn5(self.conv5(c4_spike)))
        
        # Feature Fusion
        p5 = c5_spike
        p4 = self.upsample(c3_spike)
        fused_feature = torch.cat([p4, p5], dim=1)
        fused_feature = self.concat_conv(fused_feature)
        
        # Detection Head
        detect_mem = detect_spike = torch.full_like(fused_feature, 0.5)
        detect_mem, detect_spike = self.LIF1(detect_mem, detect_spike, self.detect_conv(fused_feature))
        
        return detect_spike  # YOLO Output

# Model Initialization
model = EMS_YOLO(num_steps=5, num_classes=80)
print(model)
