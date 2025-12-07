import cv2
import numpy as np
import matplotlib.pyplot as plt

base_img_path = "./real/real1.jpg"   
residual_img_path = 'residual_gaussian.png'
save_path = 'merged_result.png'
alpha = 0.5  

base = cv2.imread(base_img_path)
resid = cv2.imread(residual_img_path)

if base is None or resid is None:
    raise FileNotFoundError("error")

h, w = resid.shape[:2]
base = cv2.resize(base, (w, h))

base_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB) / 255.0
resid_rgb = cv2.cvtColor(resid, cv2.COLOR_BGR2RGB) / 255.0

merged = np.clip(base_rgb + resid_rgb, 0, 1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(base_rgb); plt.title("Base Image")
plt.subplot(1,3,2); plt.imshow(resid_rgb); plt.title("Residual Image")
plt.subplot(1,3,3); plt.imshow(merged); plt.title("Merged Result")
plt.show()

merged_uint8 = (merged * 255).astype(np.uint8)
cv2.imwrite(save_path, cv2.cvtColor(merged_uint8, cv2.COLOR_RGB2BGR))