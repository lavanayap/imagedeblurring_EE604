
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Reproducible random noise
rng = np.random.default_rng(42)

# 1. Generate a synthetic fluorescence image
size = 256
image = np.zeros((size, size), dtype=float)

y, x = np.indices((size, size))

# Simulated fluorescent spots
spots = [
    (65, 75, 2.5),
    (70, 84, 2.5),
    (150, 160, 3.0),
    (185, 90, 2.0),
]

for cy, cx, radius in spots:
    image += np.exp(
        -((x - cx)**2 + (y - cy)**2)
        / (2 * radius**2)
    )

# Add a thin fluorescent filament
filament = np.abs(y - (0.45 * x + 50)) < 1.2
image += 0.65 * filament

# Normalize to [0, 1]
image = image / image.max()

# 2. Simulate optical blur using a Gaussian PSF
psf_sigma = 2.0
blurred = gaussian_filter(
    image,
    sigma=psf_sigma,
    mode="reflect"
)

# 3. Add photon-counting and sensor noise
photon_scale = 100
read_noise_sigma = 0.01

poisson_image = (
    rng.poisson(photon_scale * blurred)
    / photon_scale
)

noisy_image = poisson_image + rng.normal(
    0, read_noise_sigma, image.shape
)

# 4. Display the results
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

titles = [
    "Original Ground Truth",
    "Blurred Image",
    "Blurred + Noisy Image"
]

images = [image, blurred, noisy_image]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("phase1_results.png", dpi=200)
plt.show()
