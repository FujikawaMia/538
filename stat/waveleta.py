import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from pytorch_wavelets import DWTForward


# -------------------- HighFreqLossGPU --------------------
class MultiScaleHF_Loss(nn.Module):
    def __init__(self, J=1, wave='db3', mode='zero', scale_weights=None):
        super(MultiScaleHF_Loss, self).__init__()
        self.J = J
        self.xfm = DWTForward(J=J, wave=wave, mode=mode)
        self.mse = nn.MSELoss()
        if scale_weights is None:
            self.scale_weights = [1.0]*J
        else:
            assert len(scale_weights) == J
            self.scale_weights = scale_weights

    def forward(self, generated, target):
        # DWT decomposition
        Yl_g, Yh_g = self.xfm(generated)
        Yl_t, Yh_t = self.xfm(target)

        total_loss = 0.0
        for j in range(self.J):
            HF_g = torch.abs(Yh_g[j]).sum(dim=2) 
            HF_t = torch.abs(Yh_t[j]).sum(dim=2)
            
            level_loss = self.mse(HF_g, HF_t)
            total_loss += self.scale_weights[j] * level_loss

        return total_loss

# ----------------------------------------
max_watermark_strength = 0.02
hf_loss_weight = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# LPIPS and HighFreqLoss
lpips_fn = lpips.LPIPS(net='alex').to(device)
# Multi-scale high-frequency loss
hf_fn = MultiScaleHF_Loss(J=3, wave='db3', scale_weights=[1.0, 1.0 , 0.5]).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

img_pil = Image.open("1.jpg").convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(device).clone().detach()

watermark = torch.randn_like(img_tensor, requires_grad=True, device=device)
optimizer = optim.Adam([watermark], lr=0.01)

for step in range(1000):
    optimizer.zero_grad()
    
    wm = max_watermark_strength * torch.tanh(watermark)
    watermarked = torch.clamp(img_tensor + wm, -1, 1)
    
    lpips_val = lpips_fn(img_tensor, watermarked)
    hf_val = hf_fn(watermarked, img_tensor)
    
    # loss
    loss = lpips_val-hf_loss_weight * hf_val
    
    if(step % 100 == 0):
        print(f"step {step}: LPIPS={lpips_val.item():.4f}, HF={hf_val.item():.4f}")
    
    loss.backward()
    optimizer.step()

final_img = watermarked.detach().cpu()[0]  
final_img = (final_img * 0.5 + 0.5).clamp(0,1)
final_pil = transforms.ToPILImage()(final_img)
final_pil.save("out.png")
