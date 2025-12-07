path_A = "./real/real1.jpg"   
path_B = "./fake/fake4.png"   
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "./fake/fake4.png"  
save_path = 'residual_gaussian.png'
sigma = 1000

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"error:{img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

denoised = cv2.GaussianBlur(img_rgb, ksize=(0, 0), sigmaX=sigma)

residual = img_rgb - denoised

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(np.clip(denoised, 0, 1))
plt.title(f"Gaussian Denoised ({sigma})")

plt.subplot(1, 3, 3)
res_norm = residual
plt.imshow(res_norm)
plt.title("Residual (Normalized)")
plt.show()

residual_uint8 = np.clip(res_norm * 255, 0, 255).astype(np.uint8)
residual_bgr = cv2.cvtColor(residual_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite(save_path, residual_bgr)


