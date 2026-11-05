
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import tifffile

from richardson_lucy import richardson_lucy_3d


# Dataset and output directories
DATA = Path("data/raw")
OUT = Path("results")

OUT.mkdir(exist_ok=True)


def load_stack(filename):

    path = DATA / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with ZipFile(path) as archive:

        names = [
            n for n in archive.namelist()
            if n.lower().endswith((".tif", ".tiff"))
            and not Path(n).name.startswith("._")
            and not n.startswith("__MACOSX/")
        ]

        if len(names) != 1:
            raise ValueError(
                f"Expected one multipage TIFF in {filename}"
            )

        with archive.open(names[0]) as file:
            volume = tifffile.imread(BytesIO(file.read()))

    return volume.astype(np.float32)


# Load low-SNR image and PSF
print("Loading EPFL dataset...")

low = load_stack("Bars-G10-P15-stack.zip")
psf = load_stack("PSF-Bars-stack.zip")

# Normalize image intensity
low = low / 65535.0

print("Image shape:", low.shape)
print("PSF shape:", psf.shape)

# Run Richardson-Lucy
print("\nStarting Richardson-Lucy Deconvolution...")

restored = richardson_lucy_3d(
    observed=low,
    psf=psf,
    iterations=30
)

# Save restored volume
output_path = OUT / "rl_low_snr_30.tif"

tifffile.imwrite(
    output_path,
    restored
)

print("\nRestoration completed!")

print("Output shape:", restored.shape)
print("Minimum:", restored.min())
print("Maximum:", restored.max())
print("Mean:", restored.mean())

print("Saved output:", output_path.resolve())
