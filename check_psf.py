
import numpy as np
import tifffile
import zipfile
import io
from pathlib import Path

# Dataset path
DATA_DIR = Path("data/raw")
PSF_ZIP = DATA_DIR / "PSF-Bars-stack.zip"

# Load PSF from ZIP
with zipfile.ZipFile(PSF_ZIP, "r") as z:
    tif_files = [
        f for f in z.namelist()
        if f.lower().endswith((".tif", ".tiff"))
    ]

    print("TIFF files found:", tif_files)

    psf = tifffile.imread(
        io.BytesIO(z.read(tif_files[0]))
    ).astype(np.float32)

# Basic information
print("\n--- PSF INFORMATION ---")
print("Shape:", psf.shape)
print("Data type:", psf.dtype)
print("Minimum:", psf.min())
print("Maximum:", psf.max())
print("Sum:", psf.sum())

# Normalize PSF
psf_normalized = psf / psf.sum()

print("\n--- NORMALIZED PSF ---")
print("Normalized sum:", psf_normalized.sum())

# Maximum intensity position
max_position = np.unravel_index(
    np.argmax(psf_normalized),
    psf_normalized.shape
)

print("\nMaximum position:", max_position)

# Center of mass
coords = np.indices(psf_normalized.shape)

center_of_mass = [
    np.sum(coords[i] * psf_normalized)
    for i in range(3)
]

geometric_center = [
    (size - 1) / 2
    for size in psf_normalized.shape
]

print("Center of mass:", center_of_mass)
print("Geometric center:", geometric_center)

print("\nPSF inspection completed.")
