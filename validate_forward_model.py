
import numpy as np
import tifffile
import zipfile
import io
from pathlib import Path
from scipy.fft import rfftn, irfftn, ifftshift

DATA_DIR = Path("data/raw")

FILES = {
    "Reference": "Bars-stack.zip",
    "Low SNR": "Bars-G10-P15-stack.zip",
    "High SNR": "Bars-G10-P30-stack.zip",
    "PSF": "PSF-Bars-stack.zip"
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
        ).astype(np.float32)


# --------------------------------------------------
# 1. Load volumes
# --------------------------------------------------

print("Loading dataset...")

reference = load_volume(FILES["Reference"])
low = load_volume(FILES["Low SNR"])
high = load_volume(FILES["High SNR"])
psf = load_volume(FILES["PSF"])

if not (reference.shape == low.shape == high.shape == psf.shape):
    raise ValueError("Volume dimensions do not match.")

# --------------------------------------------------
# 2. Normalize PSF
# --------------------------------------------------

psf /= psf.sum()

# --------------------------------------------------
# 3. Predict blurred reference using PSF
# --------------------------------------------------

print("Calculating forward convolution...")

H = rfftn(ifftshift(psf))

predicted = irfftn(
    H * rfftn(reference),
    s=reference.shape
).real.astype(np.float32)

# --------------------------------------------------
# 4. Compare predicted and actual observations
# --------------------------------------------------

def compare(name, predicted, observed):

    p = predicted.ravel()
    o = observed.ravel()

    # Fit observed approximately as alpha * predicted + beta
    p_mean = np.mean(p, dtype=np.float64)
    o_mean = np.mean(o, dtype=np.float64)

    p_centered = p - p_mean
    o_centered = o - o_mean

    variance = np.mean(p_centered ** 2, dtype=np.float64)

    if variance < 1e-12:
        raise ValueError("Predicted image has negligible variance.")

    covariance = np.mean(
        p_centered * o_centered,
        dtype=np.float64
    )

    alpha = covariance / variance
    beta = o_mean - alpha * p_mean

    fitted = alpha * p + beta

    residual = o - fitted

    correlation = covariance / np.sqrt(
        variance * np.mean(o_centered ** 2, dtype=np.float64)
    )

    rmse = np.sqrt(
        np.mean(residual ** 2, dtype=np.float64)
    )

    relative_rmse = rmse / (
        np.std(o, dtype=np.float64) + 1e-12
    )

    print(f"\n{name}")
    print("-" * 45)
    print("Fitted intensity scale (alpha):", alpha)
    print("Fitted background offset (beta):", beta)
    print("Correlation:", correlation)
    print("RMSE after fitting:", rmse)
    print("Relative RMSE:", relative_rmse)


print("\n--- FORWARD MODEL VALIDATION ---")

compare("LOW SNR", predicted, low)
compare("HIGH SNR", predicted, high)

print("\nValidation completed.")
