
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import zipfile
import io
from pathlib import Path
from matplotlib.colors import LogNorm

# --------------------------------------------------
# 1. Define dataset path
# --------------------------------------------------

PSF_ZIP = Path("data/raw/PSF-Bars-stack.zip")

# --------------------------------------------------
# 2. Load PSF from ZIP file
# --------------------------------------------------

with zipfile.ZipFile(PSF_ZIP, "r") as z:

    tif_files = [
        f for f in z.namelist()
        if f.lower().endswith((".tif", ".tiff"))
        and not f.startswith("__MACOSX/")
        and not f.split("/")[-1].startswith("._")
    ]

    if not tif_files:
        raise FileNotFoundError("No valid TIFF file found in PSF ZIP.")

    print("Loading:", tif_files[0])

    psf = tifffile.imread(
        io.BytesIO(z.read(tif_files[0]))
    ).astype(np.float32)

# --------------------------------------------------
# 3. Validate and normalize PSF
# --------------------------------------------------

if psf.ndim != 3:
    raise ValueError("PSF must be a 3D volume.")

if not np.all(np.isfinite(psf)):
    raise ValueError("PSF contains invalid values.")

if np.any(psf < 0) or psf.sum() <= 0:
    raise ValueError("PSF must be nonnegative with positive sum.")

psf /= psf.sum()

print("\nPSF Information")
print("-------------------------")
print("Shape:", psf.shape)
print("Minimum:", psf.min())
print("Maximum:", psf.max())
print("Normalized sum:", psf.sum())

# --------------------------------------------------
# 4. Extract central slices
# --------------------------------------------------

z_center = psf.shape[0] // 2
y_center = psf.shape[1] // 2
x_center = psf.shape[2] // 2

xy_plane = psf[z_center, :, :]
xz_plane = psf[:, y_center, :]
yz_plane = psf[:, :, x_center]

planes = [
    xy_plane,
    xz_plane,
    yz_plane
]

titles = [
    f"XY plane (Z={z_center})",
    f"XZ plane (Y={y_center})",
    f"YZ plane (X={x_center})"
]

# --------------------------------------------------
# 5. Plot PSF using logarithmic intensity scale
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (ax, plane, title) in enumerate(
    zip(axes, planes, titles)
):

    positive = plane[plane > 0]

    if positive.size == 0:
        raise ValueError(f"No positive PSF values in {title}")

    vmin = max(positive.max() * 1e-5, 1e-12)

    ax.imshow(
        plane,
        cmap="hot",
        norm=LogNorm(
            vmin=vmin,
            vmax=plane.max()
        ),
        origin="upper",
        aspect="auto"
    )

    ax.set_title(title)

    if i == 0:
        # XY plane: center around X=128, Y=128
        ax.set_xlim(x_center - 20, x_center + 20)
        ax.set_ylim(y_center + 20, y_center - 20)

    else:
        # XZ and YZ: horizontal axis is X/Y,
        # vertical axis is Z.
        ax.set_xlim(108, 148)
        ax.set_ylim(z_center + 20, z_center - 20)

    ax.set_xlabel("Spatial coordinate")
    ax.set_ylabel("Spatial coordinate")

plt.tight_layout()

# --------------------------------------------------
# 6. Save visualization
# --------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

output_path = RESULTS_DIR / "psf_log_visualization.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print("\nVisualization saved at:", output_path)

plt.show()
plt.close()
