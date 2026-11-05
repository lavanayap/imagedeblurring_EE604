
import numpy as np
import tifffile
from pathlib import Path

from richardson_lucy_calibrated import (
    richardson_lucy_calibrated_3d
)

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_volume(filename):
    import zipfile
    import io

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
observed = load_volume("Bars-G10-P30-stack.zip")
psf = load_volume("PSF-Bars-stack.zip")

observed /= 65535.0

# Diagnostic parameters
alpha = 27.296798371491153
beta = 205.1966085988679 / 65535.0

iteration_values = [80, 100, 120]

for iterations in iteration_values:

    print(f"\nRunning {iterations} iterations")

    restored = richardson_lucy_calibrated_3d(
        observed=observed,
        psf=psf,
        alpha=alpha,
        beta=beta,
        iterations=iterations
    )

    output_path = (
        RESULTS_DIR /
        f"calibrated_rl_high_{iterations}.tif"
    )

    tifffile.imwrite(output_path, restored)

    print("Saved:", output_path)
    print("Maximum:", restored.max())
    print("Mean:", restored.mean())

print("\nAll experiments completed.")
