
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

def load_volume(filename):
    with zipfile.ZipFile(DATA_DIR / filename, "r") as z:

        tif_files = [
            f for f in z.namelist()
            if f.lower().endswith((".tif", ".tiff"))
            and not f.startswith("__MACOSX/")
            and not f.split("/")[-1].startswith("._")
        ]

        if not tif_files:
            raise FileNotFoundError(filename)

        return tifffile.imread(
            io.BytesIO(z.read(tif_files[0]))
        )

# Load volumes
reference = load_volume(FILES["Reference"]).astype(np.float32)
low = load_volume(FILES["Low SNR"]).astype(np.float32)
high = load_volume(FILES["High SNR"]).astype(np.float32)

# Identify background and signal using reference
background_mask = reference == 0
signal_mask = reference > 0

print("\n--- REGION STATISTICS ---")

print("\nReference distribution")
print("----------------------")
print("Total voxels:", reference.size)
print("Background voxels:", np.sum(background_mask))
print("Signal voxels:", np.sum(signal_mask))
print("Background percentage:",
      100 * np.mean(background_mask))
print("Signal percentage:",
      100 * np.mean(signal_mask))

def report(name, volume):

    background = volume[background_mask]
    signal = volume[signal_mask]

    print(f"\n{name}")
    print("-" * 45)

    print("BACKGROUND:")
    print("Mean:", background.mean())
    print("Std:", background.std())
    print("Median:", np.median(background))
    print("95th percentile:", np.percentile(background, 95))
    print("99th percentile:", np.percentile(background, 99))

    print("\nSIGNAL:")
    print("Mean:", signal.mean())
    print("Std:", signal.std())
    print("Median:", np.median(signal))
    print("95th percentile:", np.percentile(signal, 95))
    print("99th percentile:", np.percentile(signal, 99))

report("LOW SNR", low)
report("HIGH SNR", high)

print("\nAnalysis completed.")
