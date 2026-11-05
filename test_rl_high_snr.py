
import numpy as np
import tifffile
import zipfile
import io
from pathlib import Path

from richardson_lucy import richardson_lucy_3d

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)


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
        ).astype(np.float32)


# Load high-SNR observation and PSF
high = load_volume("Bars-G10-P30-stack.zip")
psf = load_volume("PSF-Bars-stack.zip")

# Normalize observation
high /= 65535.0

print("Input shape:", high.shape)
print("PSF shape:", psf.shape)

# Run RL
for iterations in [10, 20, 30]:

    print(f"\nRunning RL with {iterations} iterations...")

    restored = richardson_lucy_3d(
        observed=high,
        psf=psf,
        iterations=iterations
    )

    output_path = (
        RESULTS_DIR / f"rl_high_snr_{iterations}.tif"
    )

    tifffile.imwrite(output_path, restored)

    print("Output:", output_path)
    print("Minimum:", restored.min())
    print("Maximum:", restored.max())
    print("Mean:", restored.mean())

print("\nHigh-SNR RL experiments completed.")
