
from pathlib import Path
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wiener"))

from wiener_final import load_zip_volume, center_crop

DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "streamlit_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "Bars-G10-P15-stack.zip": "low_noise_sample.tif",
    "Bars-G10-P30-stack.zip": "high_noise_sample.tif",
}

shape = (64, 192, 192)

for source, destination in files.items():
    volume = load_zip_volume(DATA_DIR / source)

    # Match the same normalization and crop used
    # during dataset parameter calibration.
    volume = center_crop(
        volume / 65535.0,
        shape
    ).astype(np.float32)

    output_path = OUTPUT_DIR / destination

    tifffile.imwrite(output_path, volume)

    print(f"Saved: {output_path}")
    print(f"Shape: {volume.shape}")
    print(f"Intensity range: {volume.min():.4f} to {volume.max():.4f}")
