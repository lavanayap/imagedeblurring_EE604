
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import re

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.fft import rfftn, irfftn, ifftshift

DATA = Path("data/raw")
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def natural_key(name):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", name)
    ]


def load_zip(filename):
    path = DATA / filename

    with ZipFile(path) as archive:
        names = sorted(
            [
                n for n in archive.namelist()
                if n.lower().endswith((".tif", ".tiff"))
                and not Path(n).name.startswith("._")
            ],
            key=natural_key
        )

        if not names:
            raise ValueError(f"No TIFF images in {filename}")

        arrays = [
            tifffile.imread(BytesIO(archive.read(n)))
            for n in names
        ]

    if len(arrays) == 1:
        volume = arrays[0]
    else:
        volume = np.concatenate(
            [
                a[np.newaxis] if a.ndim == 2 else a
                for a in arrays
            ],
            axis=0
        )

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D data, got {volume.shape}"
        )

    return volume.astype(np.float32)


def wiener_3d(observed, psf, K=0.01):
    # Normalize the microscope PSF.
    psf = psf / np.sum(psf)

    # Move the centered PSF to the FFT origin.
    psf_origin = ifftshift(psf)

    H = rfftn(psf_origin)
    G = rfftn(observed)

    # Regularized inverse in the frequency domain.
    denominator = np.abs(H) ** 2 + K
    restored_fft = (
        np.conj(H) * G / denominator
    )

    restored = irfftn(
        restored_fft,
        s=observed.shape
    )

    # Fluorescence intensity should be nonnegative.
    return np.maximum(
        restored.real, 0
    ).astype(np.float32)


print("Loading EPFL dataset...")

low = load_zip("Bars-G10-P15-stack.zip")
high = load_zip("Bars-G10-P30-stack.zip")
psf = load_zip("PSF-Bars-stack.zip")

assert low.shape == high.shape == psf.shape

print("Volume shape:", low.shape)
print("Original PSF sum:", psf.sum())

# Same intensity conversion for both degraded volumes.
low = low / 65535.0
high = high / 65535.0

# Initial regularization parameter, not yet optimized.
K = 0.01

print("Restoring low-SNR volume...")
restored_low = wiener_3d(low, psf, K)

print("Restoring high-SNR volume...")
restored_high = wiener_3d(high, psf, K)

z = low.shape[0] // 2

images = [
    low[z],
    restored_low[z],
    high[z],
    restored_high[z]
]

titles = [
    "Low SNR Input",
    "Low SNR Wiener",
    "High SNR Input",
    "High SNR Wiener"
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, img, title in zip(axes, images, titles):
    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    OUT / "wiener_comparison.png",
    dpi=200
)

tifffile.imwrite(
    OUT / "wiener_low_snr.tif",
    restored_low
)

tifffile.imwrite(
    OUT / "wiener_high_snr.tif",
    restored_high
)

print("Restoration completed.")
print("Results saved in:", OUT.resolve())

plt.show()
