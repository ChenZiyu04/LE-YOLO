import torch
import torch.nn as nn

 
try:
    from zerodce_module import ZeroDCEPlusPlusYOLO as OriginalZeroDCEModule

except ImportError:
    print("="*80)
    print("❌warning: we can not find 'zerodce_module.py' in 'ZeroDCEPlusPlusYOLO'")
    print("="*80)
    raise


class ZeroDCEPlusPlusYOLO(nn.Module):
       
    def __init__(self, c1, c2, *args):
        super().__init__()
        
        print(f"✅ succeed 'ZeroDCEPlusPlusYOLO'")
        print(f"  >accept the number c1={c1}, c2={c2}, args={args}")
        
        self.real_model = OriginalZeroDCEModule() 
    
    def forward(self, x):
        return self.real_model(x)
