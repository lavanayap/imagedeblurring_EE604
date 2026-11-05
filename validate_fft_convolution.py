
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
# 2. Construct FFT representation of PSF
# --------------------------------------------------

psf_origin = ifftshift(psf)
H = rfftn(psf_origin)

# --------------------------------------------------
# 3. Create a 3D impulse at the volume center
# --------------------------------------------------

impulse = np.zeros(shape, dtype=np.float32)

center = tuple(s // 2 for s in shape)
impulse[center] = 1.0

# --------------------------------------------------
# 4. Apply FFT convolution
# --------------------------------------------------

blurred_impulse = irfftn(
    H * rfftn(impulse),
    s=shape
).real

# --------------------------------------------------
# 5. Compare output with original PSF
# --------------------------------------------------

difference = np.abs(blurred_impulse - psf)

print("\n--- FFT CONVOLUTION VALIDATION ---")

print("Volume shape:", shape)
print("Impulse position:", center)

print("PSF sum:", psf.sum())
print("Output sum:", blurred_impulse.sum())

print("Maximum absolute difference:", difference.max())
print("Mean absolute difference:", difference.mean())

# --------------------------------------------------
# 6. Check peak position
# --------------------------------------------------

output_peak = np.unravel_index(
    np.argmax(blurred_impulse),
    shape
)

psf_peak = np.unravel_index(
    np.argmax(psf),
    shape
)

print("\nPSF peak:", psf_peak)
print("Output peak:", output_peak)

# --------------------------------------------------
# 7. Check numerical agreement
# --------------------------------------------------

if np.allclose(blurred_impulse, psf, atol=1e-6):
    print("\nPASS: FFT centering convention is consistent.")
else:
    print("\nWARNING: FFT output differs from the centered PSF.")

print("\nValidation completed.")
