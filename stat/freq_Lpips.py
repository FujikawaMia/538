import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_A = "./real/real1.jpg"   
path_B = "./fake/fake4.png"   
iters = 1000                   
lr = 0.01                               
target_size = None           
show_every = 50              
seed = 0
max_watermark_strength = 0.02

torch.manual_seed(seed)
np.random.seed(seed)

def pil_to_tensor(img: Image.Image, size=None):
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BICUBIC)  # size: (H,W)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2: 
        arr = np.stack([arr]*3, axis=-1)
    # HWC -> CHW
    t = torch.from_numpy(arr).permute(2,0,1)
    return t

def tensor_to_pil(t: torch.Tensor):
    t = t.detach().clamp(0,1).cpu()
    arr = (t.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def fft_log_mag(img_t: torch.Tensor):
    Fimg = torch.fft.fft2(img_t, dim=(-2,-1))
    Fimg = torch.fft.fftshift(Fimg, dim=(-2,-1))
    mag = torch.abs(Fimg)
    logmag = torch.log1p(mag)

    B, C, H, W = logmag.shape
    logmag_flat = logmag.view(B*C, -1)
    minv = logmag_flat.min(dim=1, keepdim=True)[0]
    maxv = logmag_flat.max(dim=1, keepdim=True)[0]
    norm = ((logmag_flat - minv) / (maxv - minv + 1e-8)).view(B, C, H, W)
    return norm

def show_img_and_spectrum(title, img01):
    """ img01: (C,H,W) in [0,1] """
    with torch.no_grad():
        spec = fft_log_mag(img01.unsqueeze(0)).squeeze(0)  # (C,H,W) in [0,1]
    img_np = img01.permute(1,2,0).cpu().numpy()
    spec_np = spec.mean(0).cpu().numpy() 

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_np)
    plt.title(title)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(spec_np, cmap="inferno")
    plt.title(f"{title} spectrum (log |F|)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


imgA = Image.open(path_A).convert("RGB")
imgB = Image.open(path_B).convert("RGB")

if target_size is not None:
    H, W = target_size
else:
    H, W = imgA.height, imgA.width
    if (imgB.height, imgB.width) != (H, W):
        imgB = imgB.resize((W, H), Image.BICUBIC)

tA = pil_to_tensor(imgA, size=(H,W)).to(device)  # (C,H,W) in [0,1]
tB = pil_to_tensor(imgB, size=(H,W)).to(device)

# ========== LPIPS ==========
# pip install lpips
import lpips
lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

def to_lpips_range(x01):
    return x01*2.0 - 1.0

x = tA.clone().detach()
delta = torch.zeros_like(x, requires_grad=True) 

optimizer = torch.optim.Adam([delta], lr=lr)

print("Start optimizing to reduce LPIPS between spectra...")
with torch.enable_grad():
    for it in range(1, iters+1):
        optimizer.zero_grad()

        wm = max_watermark_strength * torch.tanh(delta)

        x_adv = (x + wm).clamp(0.0, 1.0)              
        SA = fft_log_mag(x_adv.unsqueeze(0))            
        SB = fft_log_mag(tB.unsqueeze(0))               

        loss_lpips = lpips_fn(to_lpips_range(SA), to_lpips_range(SB)).mean()

        loss = loss_lpips
        loss.backward()
        optimizer.step()

        if it % show_every == 0 or it == 1 or it == iters:
            print(f"[{it:04d}/{iters}] LPIPS(spectra)={loss_lpips.item():.4f}  total={loss.item():.4f}")


x_adv = (x + delta).clamp(0.0, 1.0).detach()

show_img_and_spectrum("A (original)", tA)
show_img_and_spectrum("B (reference)", tB)
show_img_and_spectrum("A+delta (optimized)", x_adv)

out_dir = "./out"
os.makedirs(out_dir, exist_ok=True)
tensor_to_pil(x_adv).save(os.path.join(out_dir, "A_optimized_3.png"))
print("Saved:", os.path.join(out_dir, "A_optimized_3.png"))
