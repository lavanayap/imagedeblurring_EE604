from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import re

import numpy as np
import csv
import tifffile  # used only for reading input data
import matplotlib.pyplot as plt

from inverse_deconvolution import InverseFilter, estimate_noise_sigma

DATA = Path("data/raw")
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def natural_key(name):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", name)]


def load_zip(filename):
    with ZipFile(DATA / filename) as archive:
        names = sorted(
            [n for n in archive.namelist()
             if n.lower().endswith((".tif", ".tiff"))
             and not Path(n).name.startswith("._")],
            key=natural_key,
        )
        if not names:
            raise ValueError(f"No TIFF images in {filename}")
        arrays = [tifffile.imread(BytesIO(archive.read(n))) for n in names]

    volume = arrays[0] if len(arrays) == 1 else np.concatenate(
        [a[np.newaxis] if a.ndim == 2 else a for a in arrays], axis=0)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D data, got {volume.shape}")
    return volume.astype(np.float32)


def main():
    print("Loading EPFL dataset...")
    low = load_zip("Bars-G10-P15-stack.zip") / 65535.0
    high = load_zip("Bars-G10-P30-stack.zip") / 65535.0
    psf = load_zip("PSF-Bars-stack.zip")
    assert low.shape == high.shape == psf.shape
    print("Volume shape:", low.shape)

    filt = InverseFilter(psf, low.shape)      # H computed once, reused below

    # ---- noise estimates (reference-free) ----
    sig_low = estimate_noise_sigma(low)
    sig_high = estimate_noise_sigma(high)
    print(f"Estimated noise sigma  low-SNR: {sig_low:.5f}  "
          f"high-SNR: {sig_high:.5f}")

    # ---- noise gain vs threshold (key curve for the adaptive strategy) ----
    thresholds = np.logspace(-4, 0, 60)
    gains = [filt.noise_gain(t) for t in thresholds]
    kept = [filt.kept_fraction(t) for t in thresholds]

    with open(OUT / "inverse_noise_gain.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "noise_gain", "kept_fraction"])
        for t, g, k in zip(thresholds, gains, kept):
            w.writerow([f"{t:.6g}", f"{g:.6g}", f"{k:.6g}"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].loglog(thresholds, gains)
    ax[0].set_xlabel("threshold (relative to max|H|)")
    ax[0].set_ylabel("noise variance gain")
    ax[0].set_title("Noise amplification")
    ax[1].semilogx(thresholds, kept)
    ax[1].set_xlabel("threshold (relative to max|H|)")
    ax[1].set_ylabel("fraction of frequencies inverted")
    ax[1].set_title("Frequencies kept")
    plt.tight_layout()
    plt.savefig(OUT / "inverse_noise_gain.png", dpi=200)
    plt.close(fig)

    # ---- restoration sweep ----
    settings = [("naive", 0.0)] + [("truncated", t)
                                   for t in (0.1, 0.03, 0.01)]
    z = low.shape[0] // 2
    rows = []

    fig, axes = plt.subplots(2, len(settings) + 1, figsize=(18, 7))
    for row, (name, vol, sig) in enumerate([("low", low, sig_low),
                                            ("high", high, sig_high)]):
        axes[row, 0].imshow(vol[z], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name} SNR input")
        axes[row, 0].axis("off")

        for col, (mode, thr) in enumerate(settings, start=1):
            restored, info = filt.restore(vol, thr, mode)
            vmax = np.percentile(restored[z], 99.5) or 1.0
            axes[row, col].imshow(restored[z], cmap="gray",
                                  vmin=0, vmax=vmax)
            axes[row, col].set_title(
                f"{mode} t={thr}\ngain={info['noise_gain']:.1e}",
                fontsize=9)
            axes[row, col].axis("off")
            rows.append([name, mode, thr, info["kept_fraction"],
                         info["noise_gain"], sig,
                         float(restored.mean()), float(restored.std()),
                         float(restored.max())])

    plt.tight_layout()
    plt.savefig(OUT / "inverse_filter_test.png", dpi=200)
    plt.close(fig)

    # ---- noise-aware (adaptive) threshold selection ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (name, vol, sig) in enumerate([("low", low, sig_low),
                                          ("high", high, sig_high)]):
        # tolerate output noise of ~5% of the data's dynamic range
        target_std = 0.05 * float(vol.max())
        max_gain = (target_std / max(sig, 1e-9)) ** 2
        thr = filt.threshold_for_noise_gain(max_gain)
        restored, info = filt.restore(vol, thr, "truncated")
        print(f"[{name}] adaptive threshold={thr:.4f}  "
              f"kept={info['kept_fraction']:.3f}  "
              f"gain={info['noise_gain']:.2f}")

        rows.append([name, "truncated_adaptive", thr, info["kept_fraction"],
                     info["noise_gain"], sig,
                     float(restored.mean()), float(restored.std()),
                     float(restored.max())])

        axes[2 * i].imshow(vol[z], cmap="gray", vmin=0, vmax=1)
        axes[2 * i].set_title(f"{name} SNR input")
        axes[2 * i + 1].imshow(restored[z], cmap="gray", vmin=0,
                               vmax=np.percentile(restored[z], 99.5) or 1.0)
        axes[2 * i + 1].set_title(f"{name} adaptive t={thr:.3f}")
        axes[2 * i].axis("off")
        axes[2 * i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "inverse_adaptive_result.png", dpi=200)
    plt.close(fig)

    # ---- summary CSV ----
    with open(OUT / "inverse_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["volume", "mode", "threshold", "kept_fraction",
                    "noise_gain", "est_noise_sigma", "out_mean",
                    "out_std", "out_max"])
        w.writerows(rows)

    print("\nSaved to", OUT.resolve())
    print("  inverse_noise_gain.csv / .png")
    print("  inverse_filter_test.png")
    print("  inverse_adaptive_result.png")
    print("  inverse_summary.csv")


if __name__ == "__main__":
    main()