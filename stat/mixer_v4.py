import cv2
import numpy as np

def fourier_domain_hybrid(src, tgt, beta=0.01, alpha=1,  omega=1):
    """
    Perform a hybrid Fourier-domain fusion:
    - Only operate on the low-frequency region (center of the spectrum)
    - In that region, blend both magnitude and phase toward the target image

    Args:
        tgt   : Source image (BGR)
        src   : Target image (BGR)
        beta  : Ratio controlling low-frequency region size (0-1)
        alpha : Blending strength (0 = keep tgt, 1 = fully target)

    Returns:
        blended (uint8): Resulting fused image
    """
    # Ensure both images have the same size
    # h, w = tgt.shape[:2]
    # src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    h, w = src.shape[:2]
    tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] for numerical stability
    tgt = tgt.astype(np.float32) / 255.
    src = src.astype(np.float32) / 255.

    blended = np.zeros_like(tgt)

    for c in range(3):  # process each color channel independently
        # Compute 2D Fourier transforms
        F_src = np.fft.fftshift(np.fft.fft2(src[..., c]))
        F_tgt = np.fft.fftshift(np.fft.fft2(tgt[..., c]))

        # Separate magnitude and phase
        A_src, P_src = np.abs(F_src), np.angle(F_src)
        A_tgt, P_tgt = np.abs(F_tgt), np.angle(F_tgt)

        # Define the central low-frequency region
        b = int(np.floor(min(h, w) * beta / 2))
        c_h, c_w = h // 2, w // 2

        # Start from source values
        A_new, P_new = np.copy(A_tgt), np.copy(P_tgt)

        # Blend both magnitude and phase within the low-frequency region
        A_new[c_h-b:c_h+b, c_w-b:c_w+b] = \
            (1 - alpha) * A_src[c_h-b:c_h+b, c_w-b:c_w+b] + alpha * A_tgt[c_h-b:c_h+b, c_w-b:c_w+b]
        P_new[c_h-b:c_h+b, c_w-b:c_w+b] = \
            (1 - omega) * P_src[c_h-b:c_h+b, c_w-b:c_w+b] + omega * P_tgt[c_h-b:c_h+b, c_w-b:c_w+b]

        # Recombine magnitude and phase into a complex spectrum
        F_new = A_new * np.exp(1j * P_new)

        # Inverse Fourier transform to spatial domain
        blended[..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(F_new)))

    # Clip and rescale back to [0,255]
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


# ==== Example usage ====
path_A = "./real/real4.jpg"   # target image (style / lighting)
path_B = "./fake/fake4.png"   # source image (structure)

src = cv2.imread(path_A)
tgt = cv2.imread(path_B)

# beta controls low-frequency region size; alpha controls blending strength
res = fourier_domain_hybrid(src,tgt, beta=0.1, alpha=0, omega=0)
cv2.imwrite("mixed_hybrid444.png", res)
