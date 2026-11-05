
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


DATA = Path("data/raw")
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def load_stack(filename):
    """Load a multipage TIFF volume from a ZIP file."""

    with ZipFile(DATA / filename) as archive:

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


print("Loading reference and degraded volumes...")

reference = load_stack("Bars-stack.zip") / 65535.0
degraded = load_stack("Bars-G10-P15-stack.zip") / 65535.0

restored = tifffile.imread(
    OUT / "rl_low_snr_30.tif"
).astype(np.float32)

# Verify dimensions
assert reference.shape == degraded.shape == restored.shape

print("All volumes:", reference.shape)

# Inspect intensity distribution
print("\nIntensity statistics")
print("Reference maximum:", reference.max())
print("Degraded maximum:", degraded.max())
print("Restored maximum:", restored.max())
print("Restored 99th percentile:", np.percentile(restored, 99))
print("Restored 99.9th percentile:", np.percentile(restored, 99.9))

overshoot = np.mean(restored > 1.0) * 100
print("Restored voxels above 1:", f"{overshoot:.4f}%")

# Clip only for bounded-range quality metrics.
# Keep the original restored volume unchanged.
restored_clipped = np.clip(restored, 0, 1)

# Calculate PSNR on complete 3D volumes
psnr_degraded = peak_signal_noise_ratio(
    reference, degraded, data_range=1.0
)

psnr_restored = peak_signal_noise_ratio(
    reference, restored_clipped, data_range=1.0
)

# Calculate SSIM slice by slice, then average
ssim_degraded = []
ssim_restored = []

for z in range(reference.shape[0]):

    ssim_degraded.append(
        structural_similarity(
            reference[z],
            degraded[z],
            data_range=1.0
        )
    )

    ssim_restored.append(
        structural_similarity(
            reference[z],
            restored_clipped[z],
            data_range=1.0
        )
    )

print("\n========== QUALITY METRICS ==========")

print(f"Degraded PSNR : {psnr_degraded:.4f} dB")
print(f"RL PSNR       : {psnr_restored:.4f} dB")

print(f"Degraded SSIM : {np.mean(ssim_degraded):.4f}")
print(f"RL SSIM       : {np.mean(ssim_restored):.4f}")

print("=====================================")

# Visualize the central Z slice
z = reference.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [
    reference[z],
    degraded[z],
    restored_clipped[z]
]

titles = [
    "Original Ground Truth",
    "Low-SNR Degraded",
    "Richardson-Lucy (30 iterations)"
]

# Common display scale for fair visual comparison
display_max = max(
    np.percentile(reference[z], 99.8),
    1e-6
)

for ax, img, title in zip(axes, images, titles):

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=display_max
    )

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUT / "rl_visual_comparison.png",
    dpi=200
)

plt.show()

print("\nSaved visualization:")
print(OUT / "rl_visual_comparison.png")
