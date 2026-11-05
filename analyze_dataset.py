
import numpy as np
import tifffile
import zipfile
import io
from pathlib import Path

DATA_DIR = Path("data/raw")

FILES = {
    "Reference": "Bars-stack.zip",
    "Low SNR": "Bars-G10-P15-stack.zip",
    "High SNR": "Bars-G10-P30-stack.zip"
}

def load_volume(zip_name):
    with zipfile.ZipFile(DATA_DIR / zip_name, "r") as z:
        tif_files = [
            f for f in z.namelist()
            if f.lower().endswith((".tif", ".tiff"))
            and not f.startswith("__MACOSX/")
            and not f.split("/")[-1].startswith("._")
        ]

        return tifffile.imread(
            io.BytesIO(z.read(tif_files[0]))
        )

print("\n--- DATASET INTENSITY ANALYSIS ---")

for name, filename in FILES.items():

    volume = load_volume(filename)

    values = volume.astype(np.float64)

    print(f"\n{name}")
    print("-" * 40)
    print("Shape:", volume.shape)
    print("Data type:", volume.dtype)
    print("Minimum:", values.min())
    print("Maximum:", values.max())
    print("Mean:", values.mean())
    print("Standard deviation:", values.std())
    print("Median:", np.median(values))
    print("99th percentile:", np.percentile(values, 99))
    print("99.9th percentile:", np.percentile(values, 99.9))

print("\nAnalysis completed.")
