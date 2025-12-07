import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import lpips
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_image_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def compute_ssim(img1_bgr, img2_bgr):
    if img1_bgr.shape != img2_bgr.shape:
        raise ValueError("Images must have the same size for SSIM!")

    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    img1_gray = img1_gray.astype(np.float32)
    img2_gray = img2_gray.astype(np.float32)

    score, _ = ssim(
        img1_gray,
        img2_gray,
        data_range=255,
        full=True
    )
    return float(score)


def prepare_for_lpips(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0  # 0~1

    img_chw = np.transpose(img_rgb, (2, 0, 1))

    tensor = torch.from_numpy(img_chw).unsqueeze(0)  # (1, 3, H, W)

    tensor = tensor * 2.0 - 1.0
    return tensor


def compute_lpips(img1_bgr, img2_bgr, net='alex', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = lpips.LPIPS(net=net).to(device)

    t1 = prepare_for_lpips(img1_bgr).to(device)
    t2 = prepare_for_lpips(img2_bgr).to(device)

    with torch.no_grad():
        dist = loss_fn(t1, t2)

    return float(dist.item())


def main(img_path_1, img_path_2):
    img1 = load_image_bgr(img_path_1)
    img2 = load_image_bgr(img_path_2)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    ssim_score = compute_ssim(img1, img2)
    lpips_score = compute_lpips(img1, img2, net='alex')

    print(f"SSIM:  {ssim_score:.6f}  ")
    print(f"LPIPS: {lpips_score:.6f} ")


if __name__ == "__main__":
    img_path_1 = "mixed_hybrid222.png"
    img_path_2 = "mixed_hybrid222.png"
    main(img_path_1, img_path_2)
