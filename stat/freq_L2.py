import os
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

# Optimize
iters = 1000
base_lr = 0.02
eps = 0.02       
s   = 1.0
lambda_l2 = 1e-4
lambda_tv = 1e-5
show_every = 50
seed = 0
target_size = None  

# Angular bins
ntheta = 72        # 

# Loss weights
w_ang  = 1.0       # 
w_rad  = 0.15      #
w_glob = 0.1    

torch.manual_seed(seed)
np.random.seed(seed)

# =========================
# Utils
# =========================
def pil_to_tensor(img: Image.Image, size=None):
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return torch.from_numpy(arr).permute(2,0,1)  # CHW

def tensor_to_pil(t: torch.Tensor):
    t = t.detach().clamp(0,1).cpu()
    arr = (t.permute(1,2,0).numpy()*255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def fft_logmag_raw(img_t: torch.Tensor):
    """log1p(|FFT|) with fftshift; img_t: (B,C,H,W) in [0,1]"""
    Fimg = torch.fft.fft2(img_t, dim=(-2,-1))
    Fimg = torch.fft.fftshift(Fimg, dim=(-2,-1))
    mag  = torch.abs(Fimg)
    return torch.log1p(mag)

def tv_loss(x):  # x: (B,C,H,W)
    dh = x[:,:,1:,:] - x[:,:,:-1,:]
    dw = x[:,:,:,1:] - x[:,:,:,:-1]
    return (dh.pow(2).mean() + dw.pow(2).mean())

def huber(a, b, delta=0.02):
    d = a - b
    ad = d.abs()
    return torch.where(ad <= delta, 0.5*d*d, delta*(ad - 0.5*delta))

def smooth1d(x, k=None, padding=1):
    """x: (B, L) -> light smoothing over bins to reduce jitter"""
    if k is None:
        k = torch.tensor([1., 2., 1.], device=x.device) / 4.0
    k = k.view(1,1,-1)
    # circular padding suits angles; emulate wrap-around
    x1 = torch.cat([x[:, -padding:], x, x[:, :padding]], dim=1).unsqueeze(1)
    y  = F.conv1d(x1, k, padding=0).squeeze(1)
    return y

# =========================
# Angular bin map (theta indices, no masking)
# =========================
def make_theta_bins(H, W, device, ntheta):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    cy, cx = H//2, W//2
    u = xx - cx
    v = yy - cy
    theta = torch.atan2(v, u)  # [-pi, pi]
    theta = (theta + 2*np.pi) % (2*np.pi)  # [0, 2pi)
    bin_edges = torch.linspace(0, 2*np.pi, steps=ntheta+1, device=device)
    tidx = torch.bucketize(theta.flatten(), bin_edges[1:-1])  # in [0, ntheta-1]
    tidx = tidx.view(H, W).long()
    return tidx  # (H,W) integer bin ids

def angular_profile(logmag_norm, theta_bin, ntheta):
    """
    """
    x = logmag_norm.mean(dim=1)  # (B,H,W)
    B, H, W = x.shape
    prof, cnt = [], []
    idx = theta_bin.view(-1)
    for b in range(B):
        vals = x[b].view(-1)
        acc = torch.zeros(ntheta, device=x.device)
        c   = torch.zeros(ntheta, device=x.device)
        acc.index_add_(0, idx, vals)
        c.index_add_(0, idx, torch.ones_like(vals))
        m = acc / (c + 1e-6)
        prof.append(m); cnt.append(c + 1e-6)
    return torch.stack(prof,0), torch.stack(cnt,0)  # (B,ntheta)

def radial_mean(logmag_norm, nbins=32):
    B, C, H, W = logmag_norm.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=logmag_norm.device),
        torch.arange(W, device=logmag_norm.device),
        indexing='ij'
    )
    cy, cx = H//2, W//2
    R = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    r_norm = (R / (R.max() + 1e-8)) * (nbins - 1)
    r_bin = r_norm.long().clamp(0, nbins-1)

    x = logmag_norm.mean(dim=1)  # (B,H,W)
    prof, cnt = [], []
    idx = r_bin.view(-1)
    for b in range(x.shape[0]):
        vals = x[b].view(-1)
        acc = torch.zeros(nbins, device=x.device)
        c   = torch.zeros(nbins, device=x.device)
        acc.index_add_(0, idx, vals)
        c.index_add_(0, idx, torch.ones_like(vals))
        m = acc / (c + 1e-6)
        prof.append(m); cnt.append(c + 1e-6)
    return torch.stack(prof,0), torch.stack(cnt,0)

def show_img_and_spectrum(title, img01):
    with torch.no_grad():
        spec = fft_logmag_raw(img01.unsqueeze(0)).squeeze(0)
    img_np = img01.permute(1,2,0).cpu().numpy()
    spec_np = spec.mean(0).cpu().numpy()
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img_np); plt.title(title); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(spec_np, cmap="inferno"); plt.title(f"{title} spectrum (log |F|)"); plt.axis("off")
    plt.tight_layout(); plt.show()

# =========================
# Load images
# =========================
imgA = Image.open(path_A).convert("RGB")
imgB = Image.open(path_B).convert("RGB")

if target_size is not None:
    H, W = target_size
    imgA = imgA.resize((W, H), Image.BICUBIC)
    imgB = imgB.resize((W, H), Image.BICUBIC)
else:
    H, W = imgA.height, imgA.width
    if (imgB.height, imgB.width) != (H, W):
        imgB = imgB.resize((W, H), Image.BICUBIC)

tA = pil_to_tensor(imgA, size=(H,W)).to(device)  # (C,H,W) in [0,1]
tB = pil_to_tensor(imgB, size=(H,W)).to(device)

# =========================
# Fixed normalization from B (per-channel)
# =========================
with torch.no_grad():
    logB_full = fft_logmag_raw(tB.unsqueeze(0))              # (1,C,H,W)
    muB  = logB_full.mean(dim=(-2,-1), keepdim=True)         # (1,C,1,1)
    stdB = logB_full.std (dim=(-2,-1), keepdim=True) + 1e-6  # (1,C,1,1)

def norm_with_B_stats(logX):
    return (logX - muB) / stdB

# Precompute theta bins
theta_bin = make_theta_bins(H, W, device, ntheta)

# =========================
# Tanh-parameterized delta
# =========================
u = torch.zeros_like(tA, requires_grad=True)  # unconstrained
opt = torch.optim.Adam([u], lr=base_lr)

print("Optimizing for directional (angular) stripe patterns...")
for it in range(1, iters+1):
    opt.zero_grad()

    # bounded, smooth perturbation in [-eps, eps]
    delta = eps * torch.tanh(s * u)
    x_adv = (tA + delta).clamp(0.0, 1.0)

    # spectra (no per-step minmax), normalized by B
    logA = fft_logmag_raw(x_adv.unsqueeze(0))  # (1,C,H,W)
    SA = norm_with_B_stats(logA)
    SB = norm_with_B_stats(logB_full)

    # ---------- Angular profile (main) ----------
    angA, cntA = angular_profile(SA, theta_bin, ntheta)  # (1,ntheta)
    angB, cntB = angular_profile(SB, theta_bin, ntheta)

    # Smooth over angles (circular)
    angA_s = smooth1d(angA)
    angB_s = smooth1d(angB)

    # Weights by target bin counts (normalize)
    w = cntB / cntB.sum(dim=1, keepdim=True)  # (1,ntheta)

    # Robust Huber on angular curves
    loss_ang = (huber(angA_s, angB_s, delta=0.02) * w).sum(dim=1).mean()

    # ---------- Optional small anchors ----------
    # Radial mean (small weight): helps keep overall energy distribution sane
    radA, rcA = radial_mean(SA, nbins=32)
    radB, rcB = radial_mean(SB, nbins=32)
    radA_s = smooth1d(radA); radB_s = smooth1d(radB)
    wr = rcB / rcB.sum(dim=1, keepdim=True)
    loss_rad = (huber(radA_s, radB_s, delta=0.02) * wr).sum(dim=1).mean()

    # Global per-channel mean/var (very small helper)
    loss_glob = F.mse_loss(SA.mean(dim=(-2,-1)), SB.mean(dim=(-2,-1))) + \
                F.mse_loss(SA.var (dim=(-2,-1)), SB.var (dim=(-2,-1)))


    # Total
    loss = w_ang*loss_ang + w_rad*loss_rad + w_glob*loss_glob 

    loss.backward()
    torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)
    opt.step()

    if it % show_every == 0 or it == 1 or it == iters:
        print(f"[{it:04d}/{iters}] "
              f"ang={loss_ang.item():.6f}  "
              f"rad={loss_rad.item():.6f}  "
              f"glob={loss_glob.item():.6f} "
              f"total={loss.item():.6f}")

# =========================
# Save & visualize
# =========================
x_adv = (tA + eps * torch.tanh(s * u)).clamp(0.0, 1.0).detach()

def show_img_and_spectrum(title, img01):
    with torch.no_grad():
        spec = fft_logmag_raw(img01.unsqueeze(0)).squeeze(0)
    img_np = img01.permute(1,2,0).cpu().numpy()
    spec_np = spec.mean(0).cpu().numpy()
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img_np); plt.title(title); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(spec_np, cmap="inferno"); plt.title(f"{title} spectrum (log |F|)"); plt.axis("off")
    plt.tight_layout(); plt.show()

show_img_and_spectrum("A (original)", tA)
show_img_and_spectrum("B (reference)", tB)
show_img_and_spectrum("A+delta (optimized, stripes learned)", x_adv)

out_dir = "./out"; os.makedirs(out_dir, exist_ok=True)
tensor_to_pil(x_adv).save(os.path.join(out_dir, "A_optimized_stripes.png"))
print("Saved:", os.path.join(out_dir, "A_optimized_stripes.png"))
