"""One-file calibrated 3D Richardson-Lucy workflow for EPFL Synthetic Hollow Bars.

Run from the repository root:
    python richardson_lucy_3d.py

The script searches recursively for:
    Bars-stack.zip
    Bars-G10-P15-stack.zip
    Bars-G10-P30-stack.zip
    PSF-Bars-stack.zip

IMPORTANT: alpha and beta are fitted using the reference volume. Metrics are
therefore reference-assisted evaluation results, not blind-restoration results.
The FFT operator uses circular boundary conditions.
"""
import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.fft import rfftn, irfftn, ifftshift
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

UINT16_MAX = 65535.0
EPSILON = 1e-8
DEFAULT_ITERATIONS = (30, 60, 100, 120)


def find_zip(data_dir: Path, filename: str) -> Path:
    matches = list(data_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {filename} below {data_dir}. Put the dataset ZIP files "
            "inside the repository or pass --data-dir."
        )
    if len(matches) > 1:
        print(f"Multiple copies of {filename}; using {matches[0]}")
    return matches[0]


def load_tiff_from_zip(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        names = [
            n for n in archive.namelist()
            if n.lower().endswith((".tif", ".tiff"))
            and not n.startswith("__MACOSX/")
            and not n.split("/")[-1].startswith("._")
        ]
        if not names:
            raise FileNotFoundError(f"No TIFF found inside {path}")
        if len(names) > 1:
            print(f"{path.name}: multiple TIFFs found; using {names[0]}")
        with archive.open(names[0]) as f:
            return np.asarray(tifffile.imread(f))


def load_data(data_dir: Path):
    files = {
        "reference": "Bars-stack.zip",
        "low": "Bars-G10-P15-stack.zip",
        "high": "Bars-G10-P30-stack.zip",
        "psf": "PSF-Bars-stack.zip",
    }
    raw = {key: load_tiff_from_zip(find_zip(data_dir, name))
           for key, name in files.items()}
    for key, volume in raw.items():
        print(f"{key:>9}: shape={volume.shape}, dtype={volume.dtype}")

    reference = raw["reference"].astype(np.float32) / UINT16_MAX
    low = raw["low"].astype(np.float32) / UINT16_MAX
    high = raw["high"].astype(np.float32) / UINT16_MAX
    psf = raw["psf"].astype(np.float32)

    if not (reference.shape == low.shape == high.shape == psf.shape):
        raise ValueError(
            "Reference, low-SNR, high-SNR and PSF shapes must match; got "
            f"{reference.shape}, {low.shape}, {high.shape}, {psf.shape}"
        )
    if not np.all(np.isfinite(psf)) or np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("PSF must be finite, nonnegative and have positive sum.")
    psf /= psf.sum()
    print(f"Normalized PSF sum: {psf.sum():.8f}")
    return reference, low, high, psf


def make_fft_operators(psf: np.ndarray, shape):
    transfer = rfftn(ifftshift(psf / psf.sum()))
    transfer_conjugate = np.conjugate(transfer)

    def forward(volume):
        return irfftn(transfer * rfftn(volume), s=shape).real

    def adjoint(volume):
        return irfftn(transfer_conjugate * rfftn(volume), s=shape).real

    return forward, adjoint


def fit_gain_background(blurred_reference: np.ndarray, observed: np.ndarray):
    """Least-squares fit observed ~= alpha * blurred_reference + beta."""
    x = blurred_reference.astype(np.float64).ravel()
    y = observed.astype(np.float64).ravel()
    x_mean, y_mean = x.mean(), y.mean()
    denominator = np.sum((x - x_mean) ** 2)
    if denominator <= 0:
        raise ValueError("Cannot estimate alpha: blurred reference has zero variance.")
    alpha = np.sum((x - x_mean) * (y - y_mean)) / denominator
    beta = y_mean - alpha * x_mean
    return float(max(alpha, 1e-8)), float(max(beta, 0.0))


def richardson_lucy_calibrated_3d(
    observed, psf, alpha, beta, iterations=30, epsilon=EPSILON
):
    """Calibrated 3D RL for y ~= alpha * A(x) + beta."""
    observed = np.asarray(observed, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)
    if observed.ndim != 3 or psf.ndim != 3 or observed.shape != psf.shape:
        raise ValueError("Observed image and PSF must be same-shaped 3D arrays.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(psf)):
        raise ValueError("Inputs must contain finite values.")
    if np.any(observed < 0) or np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("Observed image and PSF must be nonnegative; PSF sum > 0.")
    if alpha <= 0 or beta < 0 or iterations < 1 or epsilon <= 0:
        raise ValueError("Require alpha > 0, beta >= 0, iterations >= 1, epsilon > 0.")

    forward, adjoint = make_fft_operators(psf / psf.sum(), observed.shape)
    normalization = np.maximum(adjoint(np.ones(observed.shape, np.float32)), epsilon)
    estimate = np.maximum(observed - beta, epsilon) / alpha

    for _ in range(iterations):
        predicted = alpha * forward(estimate) + beta
        ratio = observed / np.maximum(predicted, epsilon)
        correction = adjoint(ratio) / normalization
        estimate *= np.maximum(correction, 0)
        estimate = np.maximum(estimate, 0)
    return estimate.astype(np.float32)


def calculate_metrics(reference, restored):
    mse = float(np.mean((reference - restored) ** 2))
    psnr = float(peak_signal_noise_ratio(reference, restored, data_range=1.0))
    # SSIM is the average of 2D SSIM scores over all Z slices.
    ssim = float(np.mean([
        structural_similarity(reference[z], restored[z], data_range=1.0)
        for z in range(reference.shape[0])
    ]))
    return psnr, ssim, mse


def save_volume(path: Path, volume):
    volume = np.clip(volume, 0.0, 1.0)
    tifffile.imwrite(path, np.round(volume * UINT16_MAX).astype(np.uint16))


def save_comparison(path: Path, reference, observed, restored, title):
    z = reference.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, image, label in zip(
        axes, (reference[z], observed[z], restored[z]),
        ("Reference", "Degraded", "Calibrated 3D RL")
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(data_dir: Path, output_dir: Path, iteration_counts):
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, low, high, psf = load_data(data_dir)
    forward, _ = make_fft_operators(psf, reference.shape)
    blurred_reference = forward(reference)

    rows = []
    for dataset_name, observed in (("low_snr", low), ("high_snr", high)):
        print("\n" + "=" * 65 + f"\n{dataset_name.upper()}\n" + "=" * 65)
        alpha, beta = fit_gain_background(blurred_reference, observed)
        print(f"Reference-assisted calibration: alpha={alpha:.10f}, beta={beta:.10f}")

        p, s, m = calculate_metrics(reference, observed)
        print(f"Degraded baseline: PSNR={p:.4f} dB | SSIM={s:.4f} | MSE={m:.6f}")
        rows.append({
            "dataset": dataset_name, "iterations": 0, "alpha": alpha, "beta": beta,
            "PSNR_dB": p, "SSIM_slice_average": s, "MSE": m,
            "type": "degraded_baseline"
        })

        best = None
        for count in iteration_counts:
            print(f"\nRunning {count} iterations...")
            restored = richardson_lucy_calibrated_3d(
                observed, psf, alpha, beta, iterations=count
            )
            p, s, m = calculate_metrics(reference, restored)
            print(f"PSNR={p:.4f} dB | SSIM={s:.4f} | MSE={m:.6f}")
            rows.append({
                "dataset": dataset_name, "iterations": count, "alpha": alpha, "beta": beta,
                "PSNR_dB": p, "SSIM_slice_average": s, "MSE": m,
                "type": "calibrated_rl"
            })
            if best is None or p > best["psnr"]:
                best = {"psnr": p, "ssim": s, "mse": m,
                        "iterations": count, "volume": restored}
            if count == 30:
                save_volume(output_dir / f"rl_{dataset_name}_30.tif", restored)

        if best is not None:
            best_path = output_dir / f"rl_{dataset_name}_best_{best['iterations']}.tif"
            save_volume(best_path, best["volume"])
            comparison_path = output_dir / f"{dataset_name}_visual_comparison.png"
            save_comparison(
                comparison_path, reference, observed, best["volume"],
                f"{dataset_name.replace('_', ' ').title()} — best tested: "
                f"{best['iterations']} iterations"
            )
            print(
                f"\nBest tested: {best['iterations']} iterations | "
                f"PSNR={best['psnr']:.4f} dB | SSIM={best['ssim']:.4f} | "
                f"MSE={best['mse']:.6f}"
            )
            print(f"Saved {best_path} and {comparison_path}")

    csv_path = output_dir / "rl_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("COMPLETED")
    print(f"Metrics: {csv_path}")
    print(f"Outputs: {output_dir.resolve()}")
    print("NOTE: alpha/beta were fitted using the reference volume.")
    print("These are reference-assisted evaluation results, not blind restoration.")
    print("The FFT convolution uses circular boundary conditions.")


def main():
    parser = argparse.ArgumentParser(
        description="Run calibrated 3D Richardson-Lucy on both EPFL noise levels."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("."),
        help="Search this folder recursively for the four dataset ZIP files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Folder for restored TIFFs, comparison PNGs and metrics CSV."
    )
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS),
        help="Iteration counts to evaluate (default: 30 60 100 120)."
    )
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.iterations)


if __name__ == "__main__":
    main()
