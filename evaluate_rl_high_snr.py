
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

ALPHA = 27.296798371491153
BETA = 205.1966085988679

# --------------------------------------------------
# 1. Load reference
# --------------------------------------------------

with zipfile.ZipFile(DATA_DIR / "Bars-stack.zip", "r") as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith((".tif", ".tiff"))
        and not f.startswith("__MACOSX/")
        and not f.split("/")[-1].startswith("._")
    ]

    reference = tifffile.imread(
        io.BytesIO(z.read(files[0]))
    ).astype(np.float32)

reference /= 65535.0

# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------

def evaluate(name, restored):

    # Raw reconstruction statistics
    print(f"\n{name}")
    print("-" * 50)

    print("Raw maximum:", restored.max())
    print("Raw mean:", restored.mean())

    # Convert approximate observation-domain
    # intensity to reference-domain intensity.
    #
    # This is a diagnostic calibration only.

    beta_normalized = BETA / 65535.0

    calibrated = (
        restored - beta_normalized
    ) / ALPHA

    calibrated = np.clip(calibrated, 0, 1)

    psnr = peak_signal_noise_ratio(
        reference,
        calibrated,
        data_range=1.0
    )

    # Slice-wise SSIM averaged over volume
    ssim_values = []

    for z in range(reference.shape[0]):

        score = structural_similarity(
            reference[z],
            calibrated[z],
            data_range=1.0
        )

        ssim_values.append(score)

    ssim = np.mean(ssim_values)

    mse = np.mean(
        (reference - calibrated) ** 2
    )

    print("Calibrated maximum:", calibrated.max())
    print("Calibrated mean:", calibrated.mean())
    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MSE:", mse)

    return psnr, ssim, mse


# --------------------------------------------------
# 3. Evaluate all iteration counts
# --------------------------------------------------

print("\n--- HIGH-SNR RL CALIBRATION DIAGNOSTIC ---")

for iterations in [10, 20, 30]:

    path = RESULTS_DIR / f"rl_high_snr_{iterations}.tif"

    restored = tifffile.imread(path).astype(np.float32)

    evaluate(
        f"RL - {iterations} iterations",
        restored
    )

print("\nEvaluation completed.")
