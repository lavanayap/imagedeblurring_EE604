
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import zipfile
import io
from pathlib import Path
from scipy.fft import rfftn, irfftn, ifftshift

DATA_DIR = Path("data/raw")


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


# Load data
reference = load_volume("Bars-stack.zip")
low = load_volume("Bars-G10-P15-stack.zip")
high = load_volume("Bars-G10-P30-stack.zip")
psf = load_volume("PSF-Bars-stack.zip")

# Normalize PSF
psf /= psf.sum()

# Simulate forward blur
H = rfftn(ifftshift(psf))

predicted = irfftn(
    H * rfftn(reference),
    s=reference.shape
).real

# Fit scale and offset for high-SNR visualization
p = predicted.ravel().astype(np.float64)
o = high.ravel().astype(np.float64)

alpha = np.mean((p-p.mean()) * (o-o.mean())) / np.var(p)
beta = o.mean() - alpha * p.mean()

predicted_scaled = alpha * predicted + beta

# Select central slice
z = 64

# Difference image
difference = high[z] - predicted_scaled[z]

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

images = [
    reference[z],
    predicted_scaled[z],
    high[z],
    low[z],
    difference,
    np.abs(difference)
]

titles = [
    "Reference",
    "Predicted blurred reference",
    "Actual high-SNR",
    "Actual low-SNR",
    "High-SNR signed difference",
    "High-SNR absolute difference"
]

for ax, image, title in zip(axes.flat, images, titles):
    im = ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()

Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/forward_model_comparison.png",
    dpi=150
)

plt.show()
plt.close()

print("Saved results/forward_model_comparison.png")
print("High-SNR alpha:", alpha)
print("High-SNR beta:", beta)
