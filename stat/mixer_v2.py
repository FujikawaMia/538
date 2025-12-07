import cv2
import numpy as np

def fourier_domain_mix(src, tgt, beta=0.1):
  
    h, w = src.shape[:2]
    tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)
    
    src = src.astype(np.float32) / 255.
    tgt = tgt.astype(np.float32) / 255.
    
    src_fft = np.fft.fftshift(np.fft.fft2(src, axes=(0,1)))
    tgt_fft = np.fft.fftshift(np.fft.fft2(tgt, axes=(0,1)))

    b = int(np.floor(min(h, w) * beta / 2))
    c_h, c_w = h // 2, w // 2
    
    src_fft_mix = np.copy(src_fft)
    src_fft_mix[c_h-b:c_h+b, c_w-b:c_w+b] = tgt_fft[c_h-b:c_h+b, c_w-b:c_w+b]
    
    src_ifft = np.fft.ifft2(np.fft.ifftshift(src_fft_mix), axes=(0,1))
    src_ifft = np.real(src_ifft)
    
    src_ifft = np.clip(src_ifft, 0, 1)
    return (src_ifft * 255).astype(np.uint8)


path_A = "./real/real1.jpg"   
path_B = "./fake/fake4.png"   
src = cv2.imread(path_B)
tgt = cv2.imread(path_A)

res = fourier_domain_mix(src, tgt, beta=0.1)
cv2.imwrite("mixed.jpg", res)
