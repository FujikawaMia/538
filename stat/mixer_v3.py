import numpy as np
import cv2


path_A = "./real/real1.jpg"   
path_B = "./fake/fake4.png"   

I1 = cv2.imread(path_A)
I2 = cv2.imread(path_B)

h, w = I1.shape[:2]
I2 = cv2.resize(I2, (w, h), interpolation=cv2.INTER_AREA)

I1 = I1.astype(np.float32)
I2 = I2.astype(np.float32)

alpha = 0.3

I_new = np.zeros_like(I1)
for c in range(3):
    F1 = np.fft.fftshift(np.fft.fft2(I1[..., c]))
    F2 = np.fft.fftshift(np.fft.fft2(I2[..., c]))
    
    A1, P1 = np.abs(F1), np.angle(F1)
    A2, P2 = np.abs(F2), np.angle(F2)
    
    A_new = (1 - alpha) * A1 + alpha * A2
    P_new = (1 - alpha) * P1 + alpha * P2

    F_new = A_new * np.exp(1j * P_new)
    
    I_new[..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(F_new)))

I_new = np.clip(I_new, 0, 255).astype(np.uint8)
cv2.imwrite('out_both.png', I_new)