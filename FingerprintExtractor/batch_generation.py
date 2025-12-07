import os
import subprocess

base_dir = "/home/lyon/fingerprint/data_generator/data/test"
out_dir = "/home/lyon/fingerprint/Analysis"

subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for name in subdirs:
    files_path = os.path.join(base_dir, name)
    out_name = name
    cmd = [
        "python",
        "generate_images.py",
        "--files_path", files_path,
        "--out_dir", out_dir,
        "--out_name", out_name
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
