
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from inverse.inverse_deconvolution import inverse_filter


# ==========================================
# 1. Generate synthetic fluorescence image
# ==========================================

def generate_test_image(size=256):

    image = np.zeros((size, size), dtype=np.float64)

    # Fluorescent filaments
    image[40:210, 55:60] = 0.8
    image[40:210, 110:115] = 0.8
    image[40:210, 165:170] = 0.8

    # Fluorescent circular structures
    y, x = np.ogrid[:size, :size]

    centers = [
        (70, 70),
        (120, 190),
        (190, 100),
        (180, 210)
    ]

    for cy, cx in centers:

        image += 0.9 * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * 5**2)
        )

    return np.clip(image, 0, 1)


# ==========================================
# 2. Generate normalized Gaussian PSF
# ==========================================

def generate_psf(size=15, sigma=2.0):

    ax = np.arange(-(size // 2), size // 2 + 1)

    xx, yy = np.meshgrid(ax, ax)

    psf = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    psf /= psf.sum()

    return psf


# ==========================================
# 3. Simulate image degradation
# ==========================================

def degrade_image(
    clean_image,
    psf,
    photons=1000,
    read_noise=0.002
):

    # Apply the SAME PSF supplied to the inverse filter
    blurred = convolve(
        clean_image,
        psf,
        mode="reflect"
    )

    rng = np.random.default_rng(42)

    # Poisson shot noise
    noisy = rng.poisson(
        np.clip(blurred, 0, None) * photons
    ) / photons

    # Gaussian read noise
    noisy += rng.normal(
        0,
        read_noise,
        clean_image.shape
    )

    return noisy


# ==========================================
# 4. Calculate evaluation metrics
# ==========================================

def evaluate(reference, image):

    psnr = peak_signal_noise_ratio(
        reference,
        image,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        image,
        data_range=1.0
    )

    mse = np.mean((reference - image)**2)

    return psnr, ssim, mse


# ==========================================
# 5. Main test
# ==========================================

def main():

    print("\nGenerating synthetic fluorescence image...")

    clean = generate_test_image()

    psf = generate_psf(
        size=15,
        sigma=2.0
    )

    print("Generating blurred and noisy image...")

    degraded = degrade_image(
        clean,
        psf,
        photons=1000,
        read_noise=0.002
    )

    # ======================================
    # Test inverse filter at different thresholds
    # ======================================

    thresholds = [1e-3, 1e-2, 5e-2, 1e-1]

    print("\nINVERSE FILTER TEST RESULTS")
    print("-" * 65)

    print(
        f"{'Threshold':<15}"
        f"{'PSNR (dB)':<15}"
        f"{'SSIM':<15}"
        f"{'MSE':<15}"
    )

    results = []

    for threshold in thresholds:

        restored = inverse_filter(
            blurred_image=degraded,
            psf=psf,
            relative_floor=threshold,
            clip_nonnegative=True
        )

        # Basic correctness checks
        assert restored.shape == clean.shape
        assert np.all(np.isfinite(restored))

        psnr, ssim, mse = evaluate(
            clean,
            restored
        )

        results.append(
            (threshold, restored, psnr, ssim, mse)
        )

        print(
            f"{threshold:<15.4g}"
            f"{psnr:<15.4f}"
            f"{ssim:<15.4f}"
            f"{mse:<15.6f}"
        )

    # ======================================
    # Compare against degraded image
    # ======================================

    base_psnr, base_ssim, base_mse = evaluate(
        clean,
        degraded
    )

    print("\nBASELINE: DEGRADED IMAGE")
    print(f"PSNR : {base_psnr:.4f} dB")
    print(f"SSIM : {base_ssim:.4f}")
    print(f"MSE  : {base_mse:.6f}")

    # Select best threshold for this synthetic
    # ground-truth experiment only.
    best = max(results, key=lambda item: item[2])

    best_threshold, restored, psnr, ssim, mse = best

    print("\nBEST RESULT IN THIS EXPERIMENT")
    print(f"Threshold : {best_threshold}")
    print(f"PSNR      : {psnr:.4f} dB")
    print(f"SSIM      : {ssim:.4f}")
    print(f"MSE       : {mse:.6f}")

    print("\nPSNR improvement:", round(psnr - base_psnr, 4), "dB")
    print("SSIM improvement:", round(ssim - base_ssim, 4))

    # ======================================
    # Visual comparison
    # ======================================

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10)
    )

    images = [
        clean,
        degraded,
        restored,
        np.abs(clean - restored)
    ]

    titles = [
        "Ground Truth",
        "Blurred + Noisy",
        f"Inverse Filter (floor={best_threshold})",
        "Absolute Reconstruction Error"
    ]

    for ax, img, title in zip(
        axes.ravel(),
        images,
        titles
    ):

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
        "results/inverse_filter_test.png",
        dpi=200
    )

    plt.show()

    print("\nSaved results/inverse_filter_test.png")


if __name__ == "__main__":
    main()
