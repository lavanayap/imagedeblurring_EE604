"""
FINAL CPU RL / RL-TV DECONVOLUTION (3D)
=======================================
Project: Noise-Adaptive and PSF-Robust Fluorescence Microscopy Deconvolution
Dataset: EPFL Synthetic Hollow Bars

Protocol is shared with wiener/wiener_final.py (functions are IMPORTED, not
copied): same loader, crop, PSF handling, reference-assisted affine
calibration, calibration/validation/test Y-regions and PSNR/3D-SSIM metrics.

Algorithm (Dey et al., 2006):
    f_{k+1} = f_k / (1 - lam * div(grad f_k / |grad f_k|)) * H^T( g / (H f_k) )
    lam = 0 -> plain Richardson-Lucy.

Tuning uses the VALIDATION region only. The TEST region is evaluated once,
for the frozen selected configuration.

Run from project root:
    python rl_tv/rl_tv_final.py --selftest      # unit tests (fast)
    python rl_tv/rl_tv_final.py                 # full experiment

Limitations (state these in the report):
- One specimen, spatial holdouts (not independent samples)
- Reference-assisted calibration/selection (not blind)
- Periodic boundaries (FFT convolution and np.roll gradients)
- Standard RL assumes Poisson data; calibrated data carry read noise, and
  negative values are clipped, so the likelihood is only approximate
- Denominator clip (0.5..2) is a stability guard, not part of the theory
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfftn, irfftn, ifftshift

# ---- reuse the teammate's protocol so numbers are comparable ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wiener"))
from wiener_final import (  # noqa: E402
    DATA_FILES, load_zip_volume, center_crop, normalize_psf,
    psf_peak_offset, create_regions, calibrate_observation,
    calculate_metrics, save_csv, save_json,
)

DEFAULT_LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
DEFAULT_CHECKPOINTS = [5, 10, 20, 50, 100]
WORKERS = -1


# =====================================================
# CORE ALGORITHM
# =====================================================

def make_otf(psf, shape):
    psf = normalize_psf(psf)
    return rfftn(ifftshift(psf), s=shape, workers=WORKERS)


def conv(x, H):          # H f
    return irfftn(rfftn(x, workers=WORKERS) * H, s=x.shape, workers=WORKERS)


def corr(x, H):          # H^T x (flipped PSF)
    return irfftn(rfftn(x, workers=WORKERS) * np.conj(H),
                  s=x.shape, workers=WORKERS)


def grad(u):             # forward differences, periodic
    return [np.roll(u, -1, axis=a) - u for a in range(u.ndim)]


def div(p):              # backward differences (adjoint of -grad)
    return sum(pa - np.roll(pa, 1, axis=a) for a, pa in enumerate(p))


def total_variation(u):
    gr = grad(u)
    return float(np.sum(np.sqrt(sum(a ** 2 for a in gr))))


def rl_tv_iter(g, H, lam=0.01, n_iter=50, eps=1e-8, eps_tv=1e-3,
               callback=None):
    """
    Run RL-TV. callback(k, f, rel_change) is called after every iteration.
    Returns final estimate (float32).
    """
    g = np.maximum(g.astype(np.float32), eps)
    f = np.full_like(g, g.mean())

    for k in range(1, n_iter + 1):
        ratio = g / np.maximum(conv(f, H), eps)
        data_term = corr(ratio, H)

        if lam > 0:
            gr = grad(f)
            norm = np.sqrt(sum(a ** 2 for a in gr) + eps_tv ** 2)
            tv = div([a / norm for a in gr])
            denom = np.clip(1.0 - lam * tv, 0.5, 2.0)
            f_new = f * data_term / denom
        else:
            f_new = f * data_term

        f_new = np.maximum(f_new, 0).astype(np.float32)
        rel = float(np.linalg.norm(f_new - f) / (np.linalg.norm(f) + 1e-12))
        f = f_new

        if callback:
            callback(k, f, rel)

    return f


def rl_tv(g, psf, lam=0.01, n_iter=50, callback=None):
    """Convenience wrapper with the signature the adaptive-strategy needs."""
    H = make_otf(psf, g.shape)
    return rl_tv_iter(g, H, lam, n_iter, callback=callback)


# =====================================================
# SELF-TESTS
# =====================================================

def selftest():
    from scipy.ndimage import convolve as ndconv
    rng = np.random.default_rng(0)
    n = 16
    zz, yy, xx = np.mgrid[-2:3, -2:3, -2:3]
    k = np.exp(-(zz**2 + yy**2 + xx**2) / 2.0).astype(np.float32)
    k /= k.sum()
    psf = np.zeros((n, n, n), np.float32)
    psf[n//2-2:n//2+3, n//2-2:n//2+3, n//2-2:n//2+3] = k

    truth = np.zeros((n, n, n), np.float32)
    truth[5:10, 5:10, 5:10] = 1.0
    g = ndconv(truth, k, mode="wrap")
    g = np.maximum(rng.poisson(50 * g) / 50.0, 0).astype(np.float32)

    # 1. shape / finite / non-negative
    out = rl_tv(g, psf, lam=0.01, n_iter=10)
    assert out.shape == g.shape and np.isfinite(out).all() and out.min() >= 0
    print("PASS shape/finite/nonneg")

    # 2. lam=0 equals an independent plain-RL loop
    n_it = 10
    f = np.full_like(g, np.maximum(g, 1e-8).mean())
    gg = np.maximum(g, 1e-8)
    kf = k[::-1, ::-1, ::-1]
    for _ in range(n_it):
        f = f * ndconv(gg / np.maximum(ndconv(f, k, mode="wrap"), 1e-8),
                       kf, mode="wrap")
        f = np.maximum(f, 0)
    ours = rl_tv(g, psf, lam=0.0, n_iter=n_it)
    err = np.max(np.abs(ours - f)) / (f.max() + 1e-12)
    assert err < 1e-3, err
    print(f"PASS lam=0 matches plain RL (rel err {err:.2e})")

    # 3. uniform image stays uniform
    u = np.full((n, n, n), 0.3, np.float32)
    out = rl_tv(u, psf, lam=0.05, n_iter=10)
    assert np.allclose(out, 0.3, atol=1e-4), (out.min(), out.max())
    print("PASS uniform stays uniform")

    # 4. TV of output decreases as lam increases
    tvs = [total_variation(rl_tv(g, psf, lam=l, n_iter=20))
           for l in (0.0, 0.01, 0.1)]
    assert tvs[0] > tvs[1] > tvs[2], tvs
    print("PASS TV decreases with lam:", [round(t, 2) for t in tvs])


# =====================================================
# PLOTS
# =====================================================

def save_sweep_curves(rows, input_val, best, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    lams = sorted({r["lambda"] for r in rows})
    for lam in lams:
        rr = [r for r in rows if r["lambda"] == lam]
        its = [r["iterations"] for r in rr]
        lab = "RL (lam=0)" if lam == 0 else f"lam={lam:g}"
        axes[0].plot(its, [r["validation_ssim"] for r in rr], "o-", label=lab)
        axes[1].plot(its, [r["validation_psnr_db"] for r in rr], "o-", label=lab)
    for ax, key, name in ((axes[0], "ssim", "SSIM"),
                          (axes[1], "psnr_db", "PSNR (dB)")):
        ax.axhline(input_val[key], ls="--", color="red", label="Input")
        ax.set_xscale("log")
        ax.set_xlabel("RL iterations")
        ax.set_ylabel(name)
        ax.set_title(f"Validation {name}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_profile(reference, observation, rl, rltv, path):
    z, x = reference.shape[0] // 2, reference.shape[2] // 2
    pos = np.arange(reference.shape[1])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pos, reference[z, :, x], label="Reference")
    ax.plot(pos, observation[z, :, x], label="Input")
    ax.plot(pos, rl[z, :, x], label="RL")
    ax.plot(pos, rltv[z, :, x], label="RL-TV")
    ax.set_xlabel("Y (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Central Bar Intensity Profile")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def run(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.z, args.y, args.x)
    lambdas = sorted(set(args.lambdas))
    checkpoints = sorted(set(args.checkpoints))
    n_max = max(checkpoints)

    print("Loading reference and PSF...", flush=True)
    ref_full = load_zip_volume(data_dir / DATA_FILES["reference"])
    reference = center_crop(ref_full / 65535.0, shape)
    psf = normalize_psf(center_crop(
        normalize_psf(load_zip_volume(data_dir / DATA_FILES["psf"])), shape))
    print("PSF peak offset:", psf_peak_offset(psf), flush=True)

    regions = create_regions(reference)
    H = make_otf(psf, shape)

    summary_rows, manifest_conditions = [], {}

    for condition in ["low", "high"]:
        print(f"\n=== {condition.upper()} SNR ===", flush=True)
        cdir = out_dir / condition
        cdir.mkdir(parents=True, exist_ok=True)

        obs = center_crop(
            load_zip_volume(data_dir / DATA_FILES[condition]) / 65535.0, shape)
        calibrated, slope, intercept = calibrate_observation(
            obs, reference, regions["calibration"])
        input_val = calculate_metrics(reference, calibrated, regions["validation"])
        input_test = calculate_metrics(reference, calibrated, regions["test"])

        sweep_rows, change_rows = [], []
        best = {"rl": None, "rltv": None}   # keep best volumes by validation

        for lam in lambdas:
            print(f"-- lam={lam:g}", flush=True)
            t0 = time.perf_counter()

            def cb(k, f, rel, lam=lam):
                change_rows.append({"lambda": lam, "iteration": k,
                                    "relative_change": rel})
                if k not in checkpoints:
                    return
                m = calculate_metrics(reference, f, regions["validation"])
                sweep_rows.append({
                    "lambda": lam, "iterations": k,
                    "validation_psnr_db": m["psnr_db"],
                    "validation_ssim": m["ssim"],
                    "validation_mse": m["mse"],
                    "relative_change": rel,
                })
                key = "rl" if lam == 0 else "rltv"
                cur = best[key]
                if cur is None or (m["ssim"], m["psnr_db"]) > \
                        (cur["val"]["ssim"], cur["val"]["psnr_db"]):
                    best[key] = {"lam": lam, "iters": k, "val": m,
                                 "volume": f.copy()}
                print(f"   it={k:4d} val PSNR={m['psnr_db']:.3f} "
                      f"SSIM={m['ssim']:.5f} rel_change={rel:.2e}", flush=True)

            rl_tv_iter(calibrated, H, lam=lam, n_iter=n_max, callback=cb)
            print(f"   ({time.perf_counter()-t0:.1f}s)", flush=True)

        # ---- freeze selection; evaluate TEST once ----
        results = {}
        for key, label in (("rl", "RL"), ("rltv", "RL-TV")):
            b = best[key]
            if b is None:
                continue
            test = calculate_metrics(reference, b["volume"], regions["test"])
            results[key] = (b, test)
            fname = "best_rl.tif" if key == "rl" else "best_rl_tv.tif"
            tifffile.imwrite(cdir / fname, b["volume"])
            summary_rows.append({
                "condition": condition, "method": label,
                "lambda": b["lam"], "iterations": b["iters"],
                "input_PSNR": input_test["psnr_db"],
                "test_PSNR": test["psnr_db"],
                "input_SSIM": input_test["ssim"],
                "test_SSIM": test["ssim"],
                "PSNR_gain": test["psnr_db"] - input_test["psnr_db"],
                "SSIM_gain": test["ssim"] - input_test["ssim"],
                "val_SSIM_at_selection": b["val"]["ssim"],
            })
            print(f"{label}: lam={b['lam']:g}, iters={b['iters']}, "
                  f"test PSNR={test['psnr_db']:.3f} "
                  f"SSIM={test['ssim']:.5f} "
                  f"(input {input_test['psnr_db']:.3f}/{input_test['ssim']:.5f})",
                  flush=True)

        save_csv(cdir / "validation_sweep.csv", sweep_rows)
        save_csv(cdir / "relative_change_per_iteration.csv", change_rows)
        save_sweep_curves(sweep_rows, input_val, best,
                          cdir / "validation_curves.png",
                          f"{condition.upper()} SNR: RL vs RL-TV (validation)")
        if "rl" in results and "rltv" in results:
            save_profile(reference, calibrated, best["rl"]["volume"],
                         best["rltv"]["volume"], cdir / "bar_profile.png")

        manifest_conditions[condition] = {
            "calibration": {"slope": slope, "intercept": intercept},
            **{k: {"lambda": results[k][0]["lam"],
                   "iterations": results[k][0]["iters"],
                   "validation": results[k][0]["val"],
                   "test": results[k][1]} for k in results},
            "input_validation": input_val, "input_test": input_test,
        }
        del calibrated, obs, best

    save_csv(out_dir / "rl_tv_summary.csv", summary_rows)
    save_json(out_dir / "experiment_manifest.json", {
        "project": "Fluorescence Microscopy Deconvolution",
        "algorithm": "3D RL / RL-TV (Dey et al. 2006)",
        "dataset": "EPFL Synthetic Hollow Bars",
        "volume_shape_zyx": list(shape),
        "lambdas": lambdas, "iteration_checkpoints": checkpoints,
        "eps": 1e-8, "eps_tv": 1e-3, "denominator_clip": [0.5, 2.0],
        "initial_estimate": "constant = mean(input)",
        "regions_y": {n: [r[1].start, r[1].stop] for n, r in regions.items()},
        "evaluation": {"type": "internal_spatial_holdout",
                       "selection": "validation_ssim (tie: validation_psnr)",
                       "metric_data_range": 1.0, "metric_clipping": [0, 1],
                       "ssim_dimensions": 3,
                       "calibration": "reference_assisted (shared with Wiener)"},
        "fft_boundary": "periodic",
        "conditions": manifest_conditions,
    })
    print("\nRL-TV COMPLETE. Results in:", out_dir.resolve(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="rl_tv/results_final")
    p.add_argument("--z", type=int, default=128)
    p.add_argument("--y", type=int, default=256)
    p.add_argument("--x", type=int, default=256)
    p.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    p.add_argument("--checkpoints", nargs="+", type=int,
                   default=DEFAULT_CHECKPOINTS)
    a = p.parse_args()
    selftest() if a.selftest else run(a)