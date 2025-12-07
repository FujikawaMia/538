import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = "/home/lyon/fingerprint/Analysis"
out_path = os.path.join(base_dir, "all_fft2_summary.png")

subdirs = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and 
       os.path.exists(os.path.join(base_dir, d, "fft2_gray.png"))
])

n = len(subdirs)
cols = math.ceil(math.sqrt(n))
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
axes = axes.flatten()

for i, name in enumerate(subdirs):
    img_path = os.path.join(base_dir, name, "fft2_gray.png")
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(name, fontsize=10)
    axes[i].axis("off")

for j in range(len(subdirs), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved summary image to {out_path}")
