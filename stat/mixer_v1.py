import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load color image
img = Image.open("./real/real1.jpg").convert("RGB")
arr = np.asarray(img).astype(np.float32)

h, w, _ = arr.shape
cy, cx = h//2, w//2
Y, X = np.ogrid[:h, :w]
R = np.sqrt((X-cx)**2 + (Y-cy)**2)

# Function: apply FFT manipulation to each channel
def process_channels(arr, modify_func):
    modified_F = []
    for i in range(3):
        F = np.fft.fftshift(np.fft.fft2(arr[:,:,i]))
        F_mod = modify_func(F)
        modified_F.append(F_mod)
    return modified_F

# Function: inverse FFT to reconstruct image
def recover(F_list):
    rec_channels = []
    for F in F_list:
        rec = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
        rec_channels.append(rec)
    rec = np.stack(rec_channels, axis=2)
    return np.clip(rec, 0, 255).astype(np.uint8)

# Different frequency manipulations
def lowpass(F):  # blur
    mask = np.exp(-(R/50)**2)
    return F * mask

def stripe(F):  # periodic peaks
    F2 = F.copy()
    F2[cy, cx+40] += 1e6
    F2[cy, cx-40] += 1e6
    return F2

def ring(F):  # ring-shaped boost
    ring_mask = (np.abs(R-600) < 200).astype(float) * 1.1
    return F * (1 + ring_mask)

def phase_noise(F):  # random phase perturbation
    mag = np.abs(F)
    phase = np.angle(F)
    phase += np.random.normal(0, 0.5, phase.shape)
    return mag * np.exp(1j*phase)

def ring_and_blur(F):
    # Ring boost
    ring_mask = (np.abs(R-600) < 200).astype(float) * 5.0
    # Gaussian low-pass (blur)
    mask_lowpass = np.exp(-(R/50)**2)
    return F * mask_lowpass * (1 + ring_mask)

# Apply modifications
variants = {
    "Original": [np.fft.fftshift(np.fft.fft2(arr[:,:,i])) for i in range(3)],
    "Low-pass (blur)": process_channels(arr, lowpass),
    "Stripe artifact": process_channels(arr, stripe),
    "Ring artifact": process_channels(arr, ring),
    "Phase noise": process_channels(arr, phase_noise)
}

# Visualization
for name, F_list in variants.items():
    rec = recover(F_list)
    plt.figure(figsize=(10,4))
    
    # Reconstructed image
    plt.subplot(1,2,1)
    plt.imshow(rec)
    plt.title(name)
    plt.axis("off")
    
    # Spectrum (log magnitude, averaged across channels)
    spectrum = sum(np.abs(F) for F in F_list) / 3
    plt.subplot(1,2,2)
    plt.imshow(np.log1p(spectrum), cmap="inferno")
    plt.title(f"{name} spectrum (log scale)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
