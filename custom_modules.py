import torch
import torch.nn as nn
from ultralytics.nn.modules import C3Ghost, CBAM

# 将C3Ghost和CBAM封装
class C3Ghost_CBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g1=1, e=0.5):
       
        super().__init__()
        self.c3ghost = C3Ghost(c1, c2, n, shortcut, g1, e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        x_out = self.c3ghost(x)
        return self.cbam(x_out)
