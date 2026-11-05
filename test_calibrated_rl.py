
import numpy as np
import tifffile
import zipfile
import io

from pathlib import Path
from richardson_lucy_calibrated import (
    richardson_lucy_calibrated_3d
)

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_volume(filename):
    with zipfile.ZipFile(DATA_DIR / filename, "r") as z:

        files = [
            f for f in z.namelist()
            if f.lower().endswith((".tif", ".tiff"))
            and not f.startswith("__MACOSX/")
            and not f.split("/")[-1].startswith("._")
        ]

        return tifffile.imread(
            io.BytesIO(z.read(files[0]))
        ).astype(np.float32)


# Load data
high = load_volume("Bars-G10-P30-stack.zip")
psf = load_volume("PSF-Bars-stack.zip")

# Normalize observation to [0, 1] scale
high /= 65535.0

# Diagnostic calibration parameters from
# the previous full-volume forward-model fit.
alpha = 27.296798371491153
beta = 205.1966085988679 / 65535.0

print("Alpha:", alpha)
print("Beta:", beta)

for iterations in [10, 20, 30]:

    print(f"\nRunning calibrated RL: {iterations} iterations")

    restored = richardson_lucy_calibrated_3d(
        observed=high,
        psf=psf,
        alpha=alpha,
        beta=beta,
        iterations=iterations
    )

    path = RESULTS_DIR / f"calibrated_rl_high_{iterations}.tif"

    tifffile.imwrite(path, restored)

    print("Saved:", path)
    print("Minimum:", restored.min())
    print("Maximum:", restored.max())
    print("Mean:", restored.mean())

print("\nCalibrated RL experiments completed.")
