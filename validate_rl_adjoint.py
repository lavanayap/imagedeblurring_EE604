
import numpy as np
import tifffile
import zipfile
import io
from pathlib import Path
from scipy.fft import rfftn, irfftn, ifftshift

# --------------------------------------------------
# 1. Load PSF
# --------------------------------------------------

PSF_ZIP = Path("data/raw/PSF-Bars-stack.zip")

with zipfile.ZipFile(PSF_ZIP, "r") as z:

    tif_files = [
        f for f in z.namelist()
        if f.lower().endswith((".tif", ".tiff"))
        and not f.startswith("__MACOSX/")
        and not f.split("/")[-1].startswith("._")
    ]

    psf = tifffile.imread(
        io.BytesIO(z.read(tif_files[0]))
    ).astype(np.float32)

psf /= psf.sum()

shape = psf.shape

# --------------------------------------------------
# 2. Construct forward and adjoint operators
# --------------------------------------------------

H = rfftn(ifftshift(psf))
H_conjugate = np.conjugate(H)

def forward(x):
    return irfftn(
        H * rfftn(x),
        s=shape
    ).real

def adjoint(y):
    return irfftn(
        H_conjugate * rfftn(y),
        s=shape
    ).real

# --------------------------------------------------
# 3. Generate random test volumes
# --------------------------------------------------

rng = np.random.default_rng(42)

x = rng.random(shape, dtype=np.float32)
y = rng.random(shape, dtype=np.float32)

# --------------------------------------------------
# 4. Apply forward and adjoint operators
# --------------------------------------------------

Ax = forward(x)
ATy = adjoint(y)

# --------------------------------------------------
# 5. Calculate inner products
# --------------------------------------------------

lhs = np.sum(Ax * y, dtype=np.float64)
rhs = np.sum(x * ATy, dtype=np.float64)

difference = abs(lhs - rhs)

relative_error = difference / max(
    abs(lhs),
    abs(rhs),
    1e-12
)

print("\n--- RL ADJOINT VALIDATION ---")

print("Forward inner product:", lhs)
print("Adjoint inner product:", rhs)
print("Absolute difference:", difference)
print("Relative error:", relative_error)

if relative_error < 1e-5:
    print("\nPASS: Forward and adjoint operators are consistent.")
else:
    print("\nFAIL: Check the forward and adjoint operators.")

print("\nValidation completed.")
