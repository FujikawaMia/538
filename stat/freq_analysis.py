import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def load_image(path, grayscale=True, resize_max=None):
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    if resize_max is not None:
        h, w = arr.shape[:2]
        scale = min(1.0, resize_max / max(h, w))
        if scale < 1.0:
            new_size = (int(w*scale), int(h*scale))
            arr = np.asarray(Image.fromarray((arr*255).astype(np.uint8)).resize(new_size, Image.LANCZOS)).astype(np.float32)/255.0
    return arr

def compute_power_spectrum(img_gray):
    # img_gray: 2D array (float in [0,1])
    F = np.fft.fft2(img_gray)
    Fshift = np.fft.fftshift(F)
    ps = np.abs(Fshift)**2
    return ps

def radial_profile(ps):
    h, w = ps.shape
    cy, cx = h//2, w//2
    y, x = np.indices((h, w))
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r_int = r.astype(np.int32)
    r_max = r_int.max()
    radial_mean = np.zeros(r_max+1, dtype=np.float64)
    counts = np.zeros(r_max+1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            ri = r_int[i,j]
            radial_mean[ri] += ps[i,j]
            counts[ri] += 1
    counts = np.maximum(counts, 1)
    radial_mean /= counts
    return radial_mean, r

def fit_log_log(freq, power, lo_frac=0.02, hi_frac=0.6):
    # freq: frequency axis (normalized), power: radial mean values
    valid = (freq > 0) & np.isfinite(power)
    if not np.any(valid):
        raise ValueError("No valid frequency bins for fitting.")
    log_f = np.log(freq[valid])
    log_p = np.log(power[valid])
    n = len(log_f)
    lo = max(1, int(n * lo_frac))
    hi = max(lo+1, int(n * hi_frac))
    coef = np.polyfit(log_f[lo:hi], log_p[lo:hi], 1)
    slope, intercept = coef[0], coef[1]
    return slope, intercept, valid

def plot_results(ps, freq, radial_mean, slope, intercept, out_dir=None, basename="result"):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Power Spectrum (log1p)")
    plt.imshow(np.log1p(ps), origin='lower')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Radial average of power spectrum (log-log)")
    valid = freq > 0
    plt.loglog(freq[valid], radial_mean[valid], label='radial mean')
    fit_curve = np.exp(intercept) * freq[valid]**slope
    plt.loglog(freq[valid], fit_curve, '--', label=f'fit slope={slope:.2f}')
    plt.xlabel('Normalized spatial frequency')
    plt.ylabel('Power')
    plt.legend()
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figpath = out_dir / f"{basename}_spectrum.png"
        plt.savefig(figpath, dpi=200)
        print(f"Saved figure to: {figpath}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Frequency-domain power spectrum analysis")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale (default: True)")
    parser.add_argument("--resize-max", type=int, default=None, help="Max dimension to resize for speed (set None to skip)")
    parser.add_argument("--save", type=str, default=None, help="Directory to save figures (if omitted, show interactive plot)")
    args = parser.parse_args()

    img = load_image(args.image, grayscale=args.grayscale, resize_max=args.resize_max)
    if img.ndim == 3:
        # for color image, analyze luminance approximation
        img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    else:
        img_gray = img

    ps = compute_power_spectrum(img_gray)
    radial_mean, rmap = radial_profile(ps)
    # normalized frequency axis (0..1)
    h, w = img_gray.shape
    max_r = max(w//2, h//2)
    freq = np.arange(len(radial_mean)) / float(max_r)

    try:
        slope, intercept, valid = fit_log_log(freq, radial_mean)
    except Exception as e:
        print("Fitting failed:", e)
        slope, intercept = np.nan, np.nan

    print(f"Estimated log-log slope: {slope:.4f} (power ~ freq^{slope:.4f})")
    plot_results(ps, freq, radial_mean, slope, intercept, out_dir=args.save, basename=Path(args.image).stem)

if __name__ == "__main__":
    main()
