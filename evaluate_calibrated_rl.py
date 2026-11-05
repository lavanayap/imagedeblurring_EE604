
import numpy as np
import tifffile
import zipfile
import io

from pathlib import Path
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")


# --------------------------------------------------
# 1. Load reference
# --------------------------------------------------

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


reference = load_volume("Bars-stack.zip")

# Normalize reference to [0, 1]
reference /= 65535.0


# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------

def evaluate(restored, iterations):

    # Do not apply another calibration.
    restored = np.maximum(restored, 0)

    # Calculate MSE
    mse = np.mean(
        (reference - restored) ** 2,
        dtype=np.float64
    )

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(
        reference,
        restored,
        data_range=1.0
    )

    # Calculate slice-wise SSIM
    ssim_values = []

    for z in range(reference.shape[0]):

        score = structural_similarity(
            reference[z],
            restored[z],
            data_range=1.0
        )

        ssim_values.append(score)

    ssim = np.mean(ssim_values)

    print(f"\nRL - {iterations} iterations")
    print("-" * 45)

    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MSE:", mse)

    print("Restored mean:", restored.mean())
    print("Restored maximum:", restored.max())

    return psnr, ssim, mse


# --------------------------------------------------
# 3. Evaluate all outputs
# --------------------------------------------------

print("\n--- CALIBRATED RL EVALUATION ---")

for iterations in [ 80, 100, 120]:

    path = (
        RESULTS_DIR /
        f"calibrated_rl_high_{iterations}.tif"
    )
   

    restored = tifffile.imread(path).astype(np.float32)
    

    evaluate(restored, iterations)


print("\nEvaluation completed.")
