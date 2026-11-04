
"""
FINAL CPU WIENER DECONVOLUTION
===========================================
Project:
Noise-Adaptive and PSF-Robust
Fluorescence Microscopy Deconvolution

Dataset:
EPFL Synthetic Hollow Bars

Features:
- 3D TIFF ZIP loading
- 3D PSF normalization
- Fourier-domain Wiener filtering
- Cached FFTs for CPU efficiency
- Reference-assisted calibration
- Calibration/validation/test regions
- K parameter optimization
- PSNR and SSIM evaluation
- No-restoration baseline
- TIFF, CSV, JSON and PNG outputs

Run from project root:
    python wiener/wiener_final.py

Research limitations:
- Uses one EPFL specimen with spatial holdouts
- Reference-assisted calibration is not blind
- Circular convolution boundary assumption
- Constant-K Wiener approximation
"""

import argparse
import csv
import json
import platform
import re
import sys
import time

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import scipy
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import rfftn, irfftn, ifftshift
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_FILES = {
    "reference": "Bars-stack.zip",
    "low": "Bars-G10-P15-stack.zip",
    "high": "Bars-G10-P30-stack.zip",
    "psf": "PSF-Bars-stack.zip",
}

DEFAULT_K_VALUES = [
    0.00001,
    0.00003,
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
]


# =====================================================
# DATA LOADING
# =====================================================

def natural_sort_key(value):

    return [
        int(part) if part.isdigit()
        else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def load_zip_volume(path):
    """
    Load a TIFF stack stored inside a ZIP archive.
    Returns float32 volume in (Z, Y, X) order.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    with ZipFile(path) as archive:

        filenames = sorted(
            [
                name
                for name in archive.namelist()
                if name.lower().endswith(
                    (".tif", ".tiff")
                )
                and not Path(name).name.startswith("._")
            ],
            key=natural_sort_key
        )

        if not filenames:
            raise ValueError(
                f"No TIFF images found in {path}"
            )

        images = []

        for name in filenames:

            image = tifffile.imread(
                BytesIO(archive.read(name))
            )

            if image.ndim == 2:
                image = image[np.newaxis, :, :]

            elif image.ndim != 3:
                raise ValueError(
                    f"Unexpected TIFF dimensions: "
                    f"{image.shape}"
                )

            images.append(image)

    volume = np.concatenate(images, axis=0)

    return volume.astype(np.float32)


def center_crop(volume, target_shape):
    """
    Center-crop a 3D volume.
    """

    slices = []

    for current, target in zip(
        volume.shape,
        target_shape
    ):

        if target > current:
            raise ValueError(
                "Requested crop exceeds volume size"
            )

        start = (current - target) // 2

        slices.append(
            slice(start, start + target)
        )

    return volume[tuple(slices)].copy()


# =====================================================
# PSF PROCESSING
# =====================================================

def normalize_psf(psf):

    psf = np.asarray(
        psf,
        dtype=np.float32
    )

    psf = np.maximum(psf, 0)

    total = np.sum(
        psf,
        dtype=np.float64
    )

    if total <= 0:
        raise ValueError("PSF has zero energy")

    return (psf / total).astype(np.float32)


def psf_peak_offset(psf):

    peak = np.unravel_index(
        np.argmax(psf),
        psf.shape
    )

    center = tuple(
        dimension // 2
        for dimension in psf.shape
    )

    return tuple(
        int(p - c)
        for p, c in zip(peak, center)
    )


# =====================================================
# CALIBRATION AND REGIONS
# =====================================================

def create_regions(reference):
    """
    Make three non-overlapping Y-axis regions.

    The foreground extent is identified from the
    reference solely for internal benchmark setup.

    Regions:
        calibration
        validation
        test

    These are spatial holdouts from one specimen,
    NOT independent biological test samples.
    """

    projection = np.max(reference, axis=0)

    foreground_rows = np.where(
        np.any(projection > 0.5, axis=1)
    )[0]

    if len(foreground_rows) < 24:
        raise ValueError(
            "Insufficient reference foreground "
            "for three spatial regions"
        )

    start = max(
        0,
        int(foreground_rows.min()) - 8
    )

    stop = min(
        reference.shape[1],
        int(foreground_rows.max()) + 9
    )

    edges = np.linspace(
        start,
        stop,
        4,
        dtype=int
    )

    regions = {}

    names = [
        "calibration",
        "validation",
        "test"
    ]

    for i, name in enumerate(names):

        y0 = int(edges[i])
        y1 = int(edges[i + 1])

        if y1 - y0 < 7:
            raise ValueError(
                "Evaluation region too small"
            )

        regions[name] = (
            slice(None),
            slice(y0, y1),
            slice(None)
        )

    return regions


def calibrate_observation(
    observation,
    reference,
    calibration_region
):
    """
    Affine calibration:
        calibrated = slope * observed + intercept

    Fits using calibration region only.
    This is reference-assisted and not suitable
    for reference-free deployment.
    """

    x = observation[
        calibration_region
    ].astype(np.float64).ravel()

    y = reference[
        calibration_region
    ].astype(np.float64).ravel()

    mean_x = x.mean()
    mean_y = y.mean()

    variance = np.mean(
        (x - mean_x) ** 2
    )

    covariance = np.mean(
        (x - mean_x) * (y - mean_y)
    )

    if variance < 1e-15:
        raise ValueError(
            "Calibration observation has "
            "insufficient intensity variation"
        )

    slope = max(
        0.0,
        covariance / variance
    )

    intercept = mean_y - slope * mean_x

    calibrated = (
        slope * observation + intercept
    )

    calibrated = np.maximum(
        calibrated,
        0
    )

    return (
        calibrated.astype(np.float32),
        float(slope),
        float(intercept)
    )


# =====================================================
# EVALUATION
# =====================================================

def calculate_metrics(
    reference,
    reconstruction,
    region
):
    """
    PSNR and 3D SSIM for a specified region.
    All metrics use the fixed [0, 1] data range.
    """

    ref = np.asarray(
        reference[region],
        dtype=np.float32
    )

    rec = np.asarray(
        reconstruction[region],
        dtype=np.float32
    )

    # Explicitly document clipping used for metrics.
    ref = np.clip(ref, 0, 1)
    rec = np.clip(rec, 0, 1)

    mse = float(
        np.mean(
            (
                ref.astype(np.float64)
                - rec.astype(np.float64)
            ) ** 2
        )
    )

    psnr = float(
        peak_signal_noise_ratio(
            ref,
            rec,
            data_range=1.0
        )
    )

    ssim = float(
        structural_similarity(
            ref,
            rec,
            data_range=1.0,
            win_size=7
        )
    )

    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim
    }


# =====================================================
# CPU-OPTIMIZED WIENER FILTER
# =====================================================

class Wiener3D:
    """
    3D Wiener reconstruction.

    FFTs are cached during initialization and reused
    for each K value.

    H = FFT(ifftshift(normalized_centered_psf))
    G = FFT(observation)

    F_hat = conj(H)*G / (abs(H)^2 + K)
    """

    def __init__(
        self,
        observation,
        psf
    ):

        if observation.ndim != 3:
            raise ValueError(
                "Expected 3D observation"
            )

        if observation.shape != psf.shape:
            raise ValueError(
                "PSF/image dimension mismatch"
            )

        self.shape = observation.shape

        self.psf = normalize_psf(psf)

        # Assumes the optical PSF is centered.
        psf_origin = ifftshift(self.psf)

        self.H = rfftn(
            psf_origin,
            workers=1
        )

        self.G = rfftn(
            observation.astype(np.float32),
            workers=1
        )

        self.numerator = (
            np.conj(self.H) * self.G
        )

        self.power = (
            np.abs(self.H) ** 2
        )

    def restore(self, K):

        if K <= 0 or not np.isfinite(K):
            raise ValueError(
                "K must be positive and finite"
            )

        restored_frequency = (
            self.numerator
            / (self.power + K)
        )

        restored = irfftn(
            restored_frequency,
            s=self.shape,
            workers=1
        )

        restored = np.maximum(
            restored.real,
            0
        )

        return restored.astype(np.float32)


# =====================================================
# CSV / JSON SAVING
# =====================================================

def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            allow_nan=False
        )


# =====================================================
# VISUAL COMPARISON
# =====================================================

def save_comparison(
    reference,
    observation,
    wiener,
    selected,
    output_path,
    title
):

    volumes = [
        reference,
        observation,
        wiener,
        selected
    ]

    labels = [
        "Reference",
        "Calibrated Input",
        "Best Wiener",
        "Selected Output"
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8)
    )

    for i, volume in enumerate(volumes):

        z = volume.shape[0] // 2

        center_slice = volume[z]

        projection = np.max(
            volume,
            axis=0
        )

        axes[0, i].imshow(
            center_slice,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, i].set_title(
            labels[i] + " — central XY"
        )

        axes[1, i].imshow(
            projection,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, i].set_title(
            labels[i] + " — XY MIP"
        )

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    fig.suptitle(title)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160
    )

    plt.close(fig)


# =====================================================
# VALIDATION CURVES
# =====================================================

def save_validation_curves(
    sweep_rows,
    input_metrics,
    best_k,
    output_path
):

    K = [
        row["K"]
        for row in sweep_rows
    ]

    ssim = [
        row["validation_ssim"]
        for row in sweep_rows
    ]

    psnr = [
        row["validation_psnr_db"]
        for row in sweep_rows
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    axes[0].semilogx(
        K, ssim, marker="o"
    )

    axes[0].axhline(
        input_metrics["ssim"],
        linestyle="--",
        color="red",
        label="Input"
    )

    axes[0].axvline(
        best_k,
        linestyle=":",
        color="green",
        label="Best Wiener K"
    )

    axes[0].set_xlabel("K")
    axes[0].set_ylabel("SSIM")
    axes[0].set_title("Validation SSIM")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogx(
        K, psnr, marker="o"
    )

    axes[1].axhline(
        input_metrics["psnr_db"],
        linestyle="--",
        color="red",
        label="Input"
    )

    axes[1].axvline(
        best_k,
        linestyle=":",
        color="green"
    )

    axes[1].set_xlabel("K")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160
    )

    plt.close(fig)


# =====================================================
# BAR INTENSITY PROFILE
# =====================================================

def save_bar_profile(
    reference,
    observation,
    restored,
    output_path
):

    z = reference.shape[0] // 2
    x = reference.shape[2] // 2

    positions = np.arange(
        reference.shape[1]
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        positions,
        reference[z, :, x],
        label="Reference"
    )

    ax.plot(
        positions,
        observation[z, :, x],
        label="Input"
    )

    ax.plot(
        positions,
        restored[z, :, x],
        label="Wiener"
    )

    ax.set_xlabel("Y (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Central Bar Intensity Profile")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160
    )

    plt.close(fig)


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def run(args):

    data_dir = Path(args.data_dir)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    shape = (
        args.z,
        args.y,
        args.x
    )

    K_values = sorted(
        set(args.k_values)
    )

    if not K_values or any(
        not np.isfinite(k) or k <= 0
        for k in K_values
    ):
        raise ValueError("Invalid K search range")

    print(
        "\nLoading EPFL reference...",
        flush=True
    )

    reference_full = load_zip_volume(
        data_dir / DATA_FILES["reference"]
    )

    reference = center_crop(
        reference_full / 65535.0,
        shape
    )

    print(
        "Loading EPFL PSF...",
        flush=True
    )

    psf_full = normalize_psf(
        load_zip_volume(
            data_dir / DATA_FILES["psf"]
        )
    )

    if reference_full.shape != psf_full.shape:
        raise ValueError(
            "Reference/PSF full shapes differ"
        )

    psf_cropped = center_crop(
        psf_full,
        shape
    )

    retained_energy = float(
        np.sum(
            psf_cropped,
            dtype=np.float64
        )
    )

    psf = normalize_psf(psf_cropped)

    print("Volume shape:", shape, flush=True)
    print("PSF sum:", float(psf.sum()), flush=True)
    print(
        "PSF peak offset:",
        psf_peak_offset(psf),
        flush=True
    )

    print(
        "PSF retained energy:",
        retained_energy,
        flush=True
    )

    if retained_energy < 0.95:
        print(
            "WARNING: More than 5% of PSF "
            "energy lost during cropping."
        )

    if psf_peak_offset(psf) != (0, 0, 0):
        print(
            "WARNING: PSF maximum is not "
            "at assumed center."
        )

    regions = create_regions(reference)

    summary_rows = []

    # -------------------------------------------------
    # PROCESS EACH NOISE CONDITION
    # -------------------------------------------------

    for condition in ["low", "high"]:

        print(
            f"\nProcessing {condition.upper()} SNR",
            flush=True
        )

        condition_dir = output_dir / condition

        condition_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        observed_full = load_zip_volume(
            data_dir / DATA_FILES[condition]
        )

        if observed_full.shape != reference_full.shape:
            raise ValueError(
                "EPFL volumes have different shapes"
            )

        observed = center_crop(
            observed_full / 65535.0,
            shape
        )

        del observed_full

        # Reference-assisted calibration
        calibrated, slope, intercept = (
            calibrate_observation(
                observed,
                reference,
                regions["calibration"]
            )
        )

        print(
            f"Calibration: slope={slope:.6f}, "
            f"intercept={intercept:.6f}",
            flush=True
        )

        input_validation = calculate_metrics(
            reference,
            calibrated,
            regions["validation"]
        )

        input_test = calculate_metrics(
            reference,
            calibrated,
            regions["test"]
        )

        # Initialize FFT engine once
        engine = Wiener3D(
            calibrated,
            psf
        )

        sweep_rows = []

        # ---------------------------------------------
        # K PARAMETER OPTIMIZATION
        # ---------------------------------------------

        for K in K_values:

            start_time = time.perf_counter()

            restored = engine.restore(K)

            runtime = (
                time.perf_counter() - start_time
            )

            validation = calculate_metrics(
                reference,
                restored,
                regions["validation"]
            )

            sweep_rows.append({
                "K": K,
                "validation_psnr_db":
                    validation["psnr_db"],
                "validation_ssim":
                    validation["ssim"],
                "validation_mse":
                    validation["mse"],
                "ssim_gain_vs_input":
                    validation["ssim"]
                    - input_validation["ssim"],
                "psnr_gain_vs_input":
                    validation["psnr_db"]
                    - input_validation["psnr_db"],
                "reconstruction_seconds":
                    runtime
            })

            print(
                f"K={K:.6g} | "
                f"PSNR={validation['psnr_db']:.3f} | "
                f"SSIM={validation['ssim']:.5f} | "
                f"FFT restore={runtime:.3f}s",
                flush=True
            )

            del restored

        # ---------------------------------------------
        # PARAMETER SELECTION
        # ---------------------------------------------

        best = max(
            sweep_rows,
            key=lambda row: (
                row["validation_ssim"],
                row["validation_psnr_db"],
                -row["K"]
            )
        )

        best_K = best["K"]

        best_wiener = engine.restore(
            best_K
        )

        # Keep Wiener output independently
        # of the optional no-restoration decision.
        if (
            input_validation["ssim"]
            >= best["validation_ssim"]
        ):
            selected_method = "no_restoration"
            selected_K = None
            selected_output = calibrated.copy()

        else:
            selected_method = "wiener"
            selected_K = best_K
            selected_output = best_wiener.copy()

        # ---------------------------------------------
        # HELD-OUT EVALUATION
        # ---------------------------------------------

        wiener_test = calculate_metrics(
            reference,
            best_wiener,
            regions["test"]
        )

        selected_test = calculate_metrics(
            reference,
            selected_output,
            regions["test"]
        )

        print(
            f"Best Wiener K: {best_K}",
            flush=True
        )

        print(
            f"Selected method: {selected_method}",
            flush=True
        )

        print(
            f"Input test SSIM: "
            f"{input_test['ssim']:.5f}",
            flush=True
        )

        print(
            f"Wiener test SSIM: "
            f"{wiener_test['ssim']:.5f}",
            flush=True
        )

        # ---------------------------------------------
        # SAVE OUTPUTS
        # ---------------------------------------------

        save_csv(
            condition_dir / "validation_sweep.csv",
            sweep_rows
        )

        save_validation_curves(
            sweep_rows,
            input_validation,
            best_K,
            condition_dir / "validation_curves.png"
        )

        save_comparison(
            reference,
            calibrated,
            best_wiener,
            selected_output,
            condition_dir / "comparison.png",
            f"{condition.upper()} SNR | "
            f"Best Wiener K={best_K}"
        )

        save_bar_profile(
            reference,
            calibrated,
            best_wiener,
            condition_dir / "bar_profile.png"
        )

        tifffile.imwrite(
            condition_dir / "calibrated_input.tif",
            calibrated
        )

        tifffile.imwrite(
            condition_dir / "best_wiener.tif",
            best_wiener
        )

        tifffile.imwrite(
            condition_dir / "selected_output.tif",
            selected_output
        )

        metadata = {
            "condition": condition,
            "algorithm": "3D constant-K Wiener",
            "best_wiener_k": best_K,
            "selected_method": selected_method,
            "selected_k": selected_K,
            "selection_metric": "validation_ssim",
            "calibration": {
                "method": "reference_assisted_affine",
                "slope": slope,
                "intercept": intercept
            },
            "input_validation": input_validation,
            "input_test": input_test,
            "best_wiener_test": wiener_test,
            "selected_output_test": selected_test,
            "psf_peak_offset": list(
                psf_peak_offset(psf)
            ),
            "psf_retained_energy": retained_energy,
            "limitations": [
                "Reference-assisted calibration",
                "Single specimen spatial holdouts",
                "Circular FFT boundary condition",
                "PSF centering assumed",
                "Discrete K grid search"
            ]
        }

        save_json(
            condition_dir / "metadata.json",
            metadata
        )

        summary_rows.append({
            "condition": condition,
            "best_wiener_K": best_K,
            "selected_method": selected_method,
            "selected_K": selected_K,
            "input_PSNR": input_test["psnr_db"],
            "wiener_PSNR": wiener_test["psnr_db"],
            "selected_PSNR": selected_test["psnr_db"],
            "input_SSIM": input_test["ssim"],
            "wiener_SSIM": wiener_test["ssim"],
            "selected_SSIM": selected_test["ssim"],
            "wiener_PSNR_gain":
                wiener_test["psnr_db"]
                - input_test["psnr_db"],
            "wiener_SSIM_gain":
                wiener_test["ssim"]
                - input_test["ssim"]
        })

        print(
            f"Saved results: {condition_dir}",
            flush=True
        )

        del engine
        del observed
        del calibrated
        del best_wiener
        del selected_output

    # -------------------------------------------------
    # SAVE FINAL SUMMARY
    # -------------------------------------------------

    save_csv(
        output_dir / "wiener_summary.csv",
        summary_rows
    )

    manifest = {
        "project":
            "Fluorescence Microscopy Deconvolution",
        "dataset":
            "EPFL Synthetic Hollow Bars",
        "dataset_url":
            "https://bigwww.epfl.ch/deconvolution/bars/",
        "volume_shape_zyx": list(shape),
        "k_values": K_values,
        "psf_retained_energy": retained_energy,
        "psf_peak_offset": list(
            psf_peak_offset(psf)
        ),
        "regions_y": {
            name: [
                region[1].start,
                region[1].stop
            ]
            for name, region in regions.items()
        },
        "evaluation": {
            "type": "internal_spatial_holdout",
            "metric_data_range": 1.0,
            "metric_clipping": [0, 1],
            "selection": "validation_ssim",
            "ssim_dimensions": 3,
            "calibration": "reference_assisted"
        },
        "fft_boundary": "periodic",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform()
    }

    save_json(
        output_dir / "experiment_manifest.json",
        manifest
    )

    print(
        "\nWIENER OPTIMIZATION COMPLETE",
        flush=True
    )

    print(
        "Results saved in:",
        output_dir.resolve(),
        flush=True
    )


# =====================================================
# COMMAND LINE
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data/raw"
    )

    parser.add_argument(
        "--output-dir",
        default="wiener/results_final"
    )

    parser.add_argument(
        "--z",
        type=int,
        default=64
    )

    parser.add_argument(
        "--y",
        type=int,
        default=192
    )

    parser.add_argument(
        "--x",
        type=int,
        default=192
    )

    parser.add_argument(
        "--k-values",
        nargs="+",
        type=float,
        default=DEFAULT_K_VALUES
    )

    args = parser.parse_args()

    run(args)
